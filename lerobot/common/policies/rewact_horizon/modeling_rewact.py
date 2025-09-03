#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://arxiv.org/abs/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from itertools import chain
from typing import Callable, List

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.common.policies.rewact_horizon.configuration_rewact import REWACTConfig
from lerobot.common.policies.act.lipo import ActionLiPo
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy


class REWACTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://arxiv.org/abs/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    config_class = REWACTConfig
    name = "rewact"

    def __init__(
        self,
        config: REWACTConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        gaze_loss_weight: float = 0.5,
        use_optimiser: bool = False,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.gaze_loss_weight = gaze_loss_weight
        self.num_cameras = len(config.image_features) if config.image_features else 0

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.model = ACT(config)
        if use_optimiser:
            self.optimiser = ActionLiPo()   # TODO - does not consider model action dimension
        else:
            self.optimiser = None

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)
            self._previous_actions = None

    @torch.no_grad
    def select_action(self,batch: dict[str, Tensor], force_model_run: bool = False) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        
        Returns:
            Tuple of (action, eoe_pred, reward_pred, gaze_preds) where:
            - action: The predicted action tensor
            - eoe_pred: End-of-episode prediction (None if using reward head)
            - reward_pred: Reward prediction (None if using eoe head)
            - gaze_preds: Gaze predictions (can be None)
        """
        self.eval()

        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [batch[key] for key in self.config.image_features]

        # If we are doing temporal ensembling, do online updates where we keep track of the number of actions
        # we are ensembling over.
        if self.config.temporal_ensemble_coeff is not None:
            actions, reward_preds, _, _ = self.model(batch)
            actions = self.unnormalize_outputs({"action": actions})["action"]
            action = self.temporal_ensembler.update(actions)
            if self.config.use_reward_head and reward_preds is not None:
                # reward_preds is now (B, 1, len(horizon_steps)) - take first horizon step
                reward_pred = torch.clamp(reward_preds[0, 0, 0], 0.0, 1.0)  # Clamp to [0, 1] range
                return action, None, reward_pred
            else:
                return action, None, None

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        gaze_preds = None
        current_reward_pred = None
        
        if len(self._action_queue) <= 20:
            actions, reward_preds, gaze_preds, _ = self.model(batch)
            actions = actions[:, :self.config.n_action_steps]
            
            if self.config.use_reward_head and reward_preds is not None:
                # Store the current reward prediction (take first horizon step)
                current_reward_pred = torch.clamp(reward_preds[0, 0, 0], 0.0, 1.0)  # Clamp to [0, 1] range

            # TODO(rcadene): make _forward return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            if self.optimiser is not None:
            # optimise action chunk via LiPo
                print("****** Using LiPo ******")
                solved, _ = self.optimiser.solve(actions.cpu().squeeze(0).numpy(), self._previous_actions, len_past_actions=20 if self._previous_actions is not None else 0)
                solved = solved[:60]
                self._previous_actions = solved

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
                self.reset()
                self._action_queue.extend(torch.from_numpy(solved).cuda().unsqueeze(0).transpose(0, 1))
            else:
                self.reset()
                self._action_queue.extend(actions.transpose(0, 1))
        elif force_model_run:
            # predict and throw away:
            _, reward_preds, gaze_preds, _ = self.model(batch)
            if self.config.use_reward_head and reward_preds is not None:
                current_reward_pred = torch.clamp(reward_preds[0, 0, 0], 0.0, 1.0)  # Clamp to [0, 1] range
        
        return self._action_queue.popleft(), None, current_reward_pred, gaze_preds

    @torch.no_grad
    def grade_action(self, batch: dict[str, Tensor], action_chunk: Tensor) -> Tensor:
        """Grade a proposed action chunk by predicting rewards for it.
        
        This method takes observations and a proposed action chunk, then returns the predicted
        rewards across all horizon steps. This is useful for evaluating action quality or
        for action optimization/planning scenarios.
        
        Args:
            batch: Dictionary containing observations (similar to select_action)
                - Must contain observation keys like "observation.state", "observation.images", etc.
                - Should NOT contain "action" as we're providing the action to evaluate
            action_chunk: Tensor of shape (B, chunk_size, action_dim) containing the action
                sequence to evaluate
                
        Returns:
            Tensor of shape (B, len(horizon_steps)) containing predicted rewards for each
            horizon step. Values are clamped to [0, 1] range.
        """
        if not self.config.use_reward_head:
            raise ValueError("grade_action requires use_reward_head=True in configuration")
            
        if not self.config.use_action_chunk_input:
            raise ValueError("grade_action requires use_action_chunk_input=True in configuration")
        
        self.eval()
        
        # Prepare the batch similar to select_action
        batch = dict(batch)  # Make a copy to avoid modifying the original
        batch["action"] = action_chunk  # Also add as regular action for compatibility
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        
        if self.config.image_features:
            batch["observation.images"] = [batch[key] for key in self.config.image_features]
        
        # Add the action chunk for evaluation (this will be used as encoder input)
        
        # Create action_is_pad mask for inference (all actions are valid during grading)
        batch_size, chunk_size = action_chunk.shape[:2]
        batch["action_is_pad"] = torch.zeros(batch_size, chunk_size, dtype=torch.bool, device=action_chunk.device)
        
        # Run forward pass to get reward predictions
        _, reward_preds, _, _ = self.model(batch)
        
        if reward_preds is None:
            raise RuntimeError("Model returned None for reward predictions")
        
        # reward_preds shape: (B, 1, len(horizon_steps))
        # Squeeze the middle dimension and clamp to [0, 1]
        reward_predictions = torch.clamp(reward_preds.squeeze(1), 0.0, 1.0)  # Shape: (B, len(horizon_steps))
        
        return reward_predictions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [batch[key] for key in self.config.image_features]

        if 'reward' in batch:
            current_reward = batch['reward']  # Shape: (batch_size, sequence_length) = (32, 100)
            batch_size = current_reward.shape[0]
            
            # Extract rewards at specified horizon steps
            # horizon_steps contains indices like [0, 5, 10, 15, 20, ...]
            horizon_indices = torch.tensor(self.config.horizon_steps, device=current_reward.device)
            
            # Clamp indices to ensure they don't exceed sequence length
            max_seq_len = current_reward.shape[1]
            horizon_indices = torch.clamp(horizon_indices, 0, max_seq_len - 1)
            
            # Index into the reward tensor: (batch_size, len(horizon_steps))
            reward_targets = current_reward[:, horizon_indices]  # Shape: (32, len(horizon_steps))
            
            batch['reward_horizons'] = reward_targets
        else:
            # If no reward in dataset, create dummy targets
            batch_size = self._get_batch_size(batch)
            batch['reward_horizons'] = torch.zeros(batch_size, len(self.config.horizon_steps))

        if 'gaze_annotations' in batch:
            batch_size = len(batch['gaze_annotations'])
            num_cameras = len(self.config.image_features)
            
            # Initialize tensors for all cameras
            # Assuming sequence length - you might need to adjust this based on your data structure
            # If gaze_annotations has a sequence dimension, use that; otherwise assume 1
            if isinstance(batch['gaze_annotations'][0], dict):
                seq_len = 1  # Single timestep per batch element
            else:
                seq_len = len(batch['gaze_annotations'][0])  # Multiple timesteps per batch element
            
            # Create tensors: [batch_size, seq_len, 2] for each camera
            gazes_stack = []
            gazes_mask = []
            
            for cam_idx, camera_key in enumerate(self.config.image_features):
                # Initialize tensors for this camera
                camera_gazes = torch.zeros(batch_size, seq_len, 2, dtype=torch.float32)
                camera_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
                
                # Fill in the data
                for batch_idx, elem in enumerate(batch['gaze_annotations']):
                    if seq_len == 1:
                        # Single timestep case
                        if camera_key in elem:
                            camera_gazes[batch_idx, 0, 0] = elem[camera_key]['x']
                            camera_gazes[batch_idx, 0, 1] = elem[camera_key]['y']
                            camera_mask[batch_idx, 0] = True
                        # else: remains False/zero (already initialized)
                    else:
                        # Multiple timestep case
                        for seq_idx, seq_elem in enumerate(elem):
                            if camera_key in seq_elem:
                                camera_gazes[batch_idx, seq_idx, 0] = seq_elem[camera_key]['x']
                                camera_gazes[batch_idx, seq_idx, 1] = seq_elem[camera_key]['y']
                                camera_mask[batch_idx, seq_idx] = True
                
                gazes_stack.append(camera_gazes)
                gazes_mask.append(camera_mask)
            
            batch["action.gaze"] = gazes_stack  # List of [B, T, 2] tensors
            batch["action.gaze_mask"] = gazes_mask  # List of [B, T] boolean tensors

        batch = self.normalize_targets(batch)
        
        # Apply co-training strategy if enabled
        if self.config.use_cotraining and self.training:
            batch = self._apply_cotraining_strategy(batch)
        
        actions_hat, reward_hat, gaze_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        # Action loss - only compute for samples where both action_focus_mask and use_action_mask are True
        if "use_action_mask" in batch:
            # Original action mask from training data
            original_action_mask = batch["use_action_mask"]  # (batch_size,)
            
            # Co-training action focus mask (if available)
            if "action_focus_mask" in batch:
                cotraining_action_mask = batch["action_focus_mask"]  # (batch_size,)
                # Union: only compute loss if BOTH masks are True
                final_action_mask = original_action_mask & cotraining_action_mask
            else:
                # No co-training, use original mask only
                final_action_mask = original_action_mask
            
            # Apply final mask
            action_mask = final_action_mask.unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)
            # Combine with existing padding mask
            combined_mask = (~batch["action_is_pad"].unsqueeze(-1)) & action_mask
            l1_loss = (
                F.l1_loss(batch["action"], actions_hat, reduction="none") * combined_mask
            ).mean()
        else:
            # No original action mask - check for co-training mask only
            if "action_focus_mask" in batch:
                action_mask = batch["action_focus_mask"].unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)
                combined_mask = (~batch["action_is_pad"].unsqueeze(-1)) & action_mask
                l1_loss = (
                    F.l1_loss(batch["action"], actions_hat, reduction="none") * combined_mask
                ).mean()
            else:
                # Original behavior - use all non-padded actions
                l1_loss = (
                    F.l1_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
                ).mean()
        
        # Reward loss for multiple horizon steps
        if self.config.use_reward_head and reward_hat is not None:
            # Reward prediction loss - use MSE for continuous values across all horizon steps
            if "reward_horizons" in batch:
                # Expect reward_horizons to be (B, len(horizon_steps)) for each horizon step
                reward_targets = batch["reward_horizons"]  # (B, len(horizon_steps))
                reward_preds_raw = reward_hat.squeeze(1)  # (B, len(horizon_steps))
                # Clamp predictions to [0, 1] range for loss computation
                reward_preds_clamped = torch.clamp(reward_preds_raw, 0.0, 1.0)  # (B, len(horizon_steps))
                
                # Apply reward mask for co-training if available
                if "reward_focus_mask" in batch:
                    reward_mask = batch["reward_focus_mask"].unsqueeze(-1)  # (B, 1)
                    # Only compute loss for reward-focused samples
                    if reward_mask.sum() > 0:  # Avoid division by zero
                        reward_loss_per_element = F.mse_loss(
                            reward_preds_clamped, reward_targets, reduction="none"
                        )  # (B, len(horizon_steps))
                        # Apply mask and preserve horizon differences - don't average across horizons yet
                        masked_loss = reward_loss_per_element * reward_mask  # (B, len(horizon_steps))
                        reward_loss = masked_loss.sum() / (reward_mask.sum() * reward_targets.shape[1])
                    else:
                        reward_loss = torch.tensor(0.0, device=actions_hat.device)
                else:
                    # No masking - compute loss for all samples, preserving horizon differences
                    reward_loss = F.mse_loss(
                        reward_preds_clamped,  # (batch_size, len(horizon_steps))
                        reward_targets,  # (batch_size, len(horizon_steps))
                        reduction="mean"  # This averages over batch AND horizons
                    )
                
                # Store additional debugging information
                with torch.no_grad():
                    # Before clamping
                    reward_preds_raw_stats = {
                        "reward_pred_raw_mean": reward_preds_raw.mean().item(),
                        "reward_pred_raw_std": reward_preds_raw.std().item(), 
                        "reward_pred_raw_min": reward_preds_raw.min().item(),
                        "reward_pred_raw_max": reward_preds_raw.max().item(),
                    }
                    
                    # After clamping
                    reward_debug_info = {
                        "reward_pred_mean": reward_preds_clamped.mean().item(),
                        "reward_pred_std": reward_preds_clamped.std().item(),
                        "reward_pred_min": reward_preds_clamped.min().item(),
                        "reward_pred_max": reward_preds_clamped.max().item(),
                        "reward_target_mean": reward_targets.mean().item(),
                        "reward_target_std": reward_targets.std().item(),
                        "reward_target_min": reward_targets.min().item(),
                        "reward_target_max": reward_targets.max().item(),
                        "reward_horizon_variance": reward_preds_clamped.var(dim=1).mean().item(),  # Variance across horizons
                        "reward_bias": reward_targets.mean().item() - reward_preds_clamped.mean().item(),  # Systematic bias
                        "reward_mae": F.l1_loss(reward_preds_clamped, reward_targets).item(),  # Mean absolute error
                    }
                    # Add raw prediction stats
                    reward_debug_info.update(reward_preds_raw_stats)
            else:
                reward_loss = torch.tensor(0.0, device=actions_hat.device)
                reward_debug_info = {}
        else:
            reward_loss = torch.tensor(0.0, device=actions_hat.device)
            reward_debug_info = {}

        # Gaze loss
        gaze_loss = torch.tensor(0.0, device=actions_hat.device)
        if "action.gaze" in batch and "action.gaze_mask" in batch:
            gaze_loss = self._compute_gaze_loss(
                gaze_hat, 
                batch["action.gaze"], 
                batch["action.gaze_mask"]
            )

        # Combine losses
        loss_dict = {
            "l1_loss": l1_loss.item(),
            "reward_loss_raw": reward_loss.item(),  # Raw reward loss without weighting
            "reward_loss_weighted": reward_loss.item() * self.config.reward_loss_weight,  # Weighted for comparison
            "gaze_loss": gaze_loss.item() * self.gaze_loss_weight if isinstance(gaze_loss, torch.Tensor) else 0,
        }
        
        # Add reward debugging information
        loss_dict.update(reward_debug_info)
        
        # Add co-training statistics if applicable
        if self.config.use_cotraining and self.training and "action_focus_mask" in batch:
            action_focus_count = batch["action_focus_mask"].sum().item()
            reward_focus_count = batch["reward_focus_mask"].sum().item()
            
            # Calculate actual samples used for action loss (intersection of both masks)
            if "use_action_mask" in batch:
                action_loss_samples = (batch["action_focus_mask"] & batch["use_action_mask"]).sum().item()
            else:
                action_loss_samples = action_focus_count
                
            loss_dict.update({
                "action_focus_samples": action_focus_count,
                "reward_focus_samples": reward_focus_count,
                "action_loss_samples": action_loss_samples,  # Actual samples used for action loss
                "cotraining_ratio_actual": action_focus_count / (action_focus_count + reward_focus_count) if (action_focus_count + reward_focus_count) > 0 else 0.0,
            })

        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + self.config.reward_loss_weight * reward_loss + self.gaze_loss_weight * gaze_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss + self.config.reward_loss_weight * reward_loss + self.gaze_loss_weight * gaze_loss

        return loss, loss_dict
    
    def _apply_cotraining_strategy(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply co-training strategy by randomly splitting batch into action-focused and reward-focused samples.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Modified batch with appropriate masking and training focus indicators
        """
        # Get batch size from any available tensor
        if "observation.images" in batch:
            batch_size = batch["observation.images"][0].shape[0]
        elif "observation.state" in batch:
            batch_size = batch["observation.state"].shape[0]
        elif "observation.environment_state" in batch:
            batch_size = batch["observation.environment_state"].shape[0]
        else:
            raise ValueError("Cannot determine batch size - no known observation keys found")
        
        # Create random mask for action-focused samples (True = action focus, False = reward focus)
        # Use a device-appropriate random generator for better reproducibility
        device = batch["index"].device if "index" in batch else torch.device('cpu')
        action_focus_mask = torch.rand(batch_size, device=device) < self.config.cotraining_ratio
        
        # Create training focus indicators
        batch["action_focus_mask"] = action_focus_mask  # (B,) - True for action-focused samples
        batch["reward_focus_mask"] = ~action_focus_mask  # (B,) - True for reward-focused samples
        
        # For action-focused samples: mask out action input to prevent seeing future actions
        # For reward-focused samples: provide action input for better reward prediction
        if self.config.use_action_chunk_input and "action" in batch:
            # For reward-focused samples, we want to provide the action as input to the encoder
            # For action-focused samples, we mask it out so they can't "cheat"
            action_for_encoder = batch["action"].clone()
            action_for_encoder[action_focus_mask] = 0.0  # Zero out actions for action-focused samples
            batch["action_for_encoder"] = action_for_encoder
        
        # Note: We do NOT modify the original use_action_mask - it remains unchanged
        # The action loss computation will handle the intersection of both masks
                            
        return batch
    
    def _get_batch_size(self, batch: dict[str, Tensor]) -> int:
        """Helper method to determine batch size from any available tensor in the batch."""
        if "observation.images" in batch:
            return batch["observation.images"][0].shape[0]
        elif "observation.state" in batch:
            return batch["observation.state"].shape[0]
        elif "observation.environment_state" in batch:
            return batch["observation.environment_state"].shape[0]
        elif "action" in batch:
            return batch["action"].shape[0]
        else:
            raise ValueError("Cannot determine batch size - no known keys found in batch")

    def _compute_gaze_loss(self, gaze_predictions: List[torch.Tensor], 
                          gaze_targets: List[torch.Tensor], 
                          gaze_masks: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute loss for gaze prediction across all cameras.
        
        Args:
            gaze_predictions: List of [B, T, 2] tensors, one per camera
            gaze_targets: List of [B, T, 2] tensors, one per camera  
            gaze_masks: List of [B, T] boolean masks, one per camera
            
        Returns:
            Scalar gaze loss averaged across all cameras
        """
        if len(gaze_predictions) == 0 or len(gaze_targets) == 0:
            return torch.tensor(0.0, device=gaze_predictions[0].device if gaze_predictions else torch.device('cpu'))
        
        total_loss = 0.0
        valid_cameras = 0
        
        for pred, target, mask in zip(gaze_predictions, gaze_targets, gaze_masks):
            # Ensure targets and masks are on the same device as predictions
            target = target.to(pred.device)
            mask = mask.to(pred.device)
            
            # Compute MSE loss for (x, y) coordinates
            coord_loss = F.mse_loss(pred, target, reduction='none')  # [B, T, 2]
            coord_loss = coord_loss.sum(dim=-1)  # [B, T] - sum over x,y dimensions
            
            # Apply mask to only include valid gaze labels
            masked_loss = coord_loss * mask.float()
            
            # Average over valid entries
            if mask.sum() > 0:
                camera_loss = masked_loss.sum() / mask.sum()
                total_loss += camera_loss
                valid_cameras += 1
        
        # Average across cameras (avoid division by zero)
        return total_loss / max(valid_cameras, 1)

class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://arxiv.org/abs/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[:i+1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ACT(nn.Module):
    """Action Chunking Transformer: The underlying neural network for ACTPolicy.

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    """

    def __init__(self, config: REWACTConfig):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        super().__init__()
        self.config = config
        self.num_cameras = len(config.image_features) if config.image_features else 0
        
        # All original ACT components (unchanged)
        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # Projection layer for joint-space configuration to hidden dimension.
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # Backbone for image feature extraction.
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (action), (image_feature_map_pixels)].
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        # Action chunk input projection - encodes action sequences (B, chunk_size, action_dim) -> (B, chunk_size, dim_model)
        if config.use_action_chunk_input:
            self.encoder_action_chunk_input_proj = nn.Linear(
                self.config.action_feature.shape[0], config.dim_model
            )
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        # Separate positional embedding for action chunk sequence
        if config.use_action_chunk_input:
            self.encoder_action_chunk_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])
        
        # Reward prediction head - now predicts multiple horizon rewards
        if config.use_reward_head:
            # Reward prediction head: predicts continuous values between 0 and 1 for each horizon step
            self.reward_head = nn.Sequential(
                nn.Linear(config.dim_model, config.dim_model // 2),
                nn.ReLU(),
                nn.Linear(config.dim_model // 2, config.dim_model // 4),
                nn.ReLU(),
                nn.Linear(config.dim_model // 4, len(config.horizon_steps)),  # One output per horizon step
                nn.Sigmoid(),
            )
            
            # Initialize final layer to reduce sigmoid bias
            # This shifts the initial sigmoid output to match target distribution better
            final_layer = self.reward_head[-2]  # Get the linear layer before sigmoid
            if hasattr(final_layer, 'bias') and final_layer.bias is not None:
                # Initialize bias to produce outputs closer to target mean
                # logit(target_mean) ≈ log(target_mean / (1 - target_mean))
                # Assuming target mean around 0.6, this gives positive bias
                nn.init.constant_(final_layer.bias, 0.4)  # Adjust this based on your target distribution

        # Gaze prediction heads - one per camera
        # Each head predicts (x, y) coordinates in [0, 1] normalized space
        self.gaze_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.dim_model, config.dim_model // 4),  # Smaller than action head
                nn.ReLU(),
                nn.Linear(config.dim_model // 4, 2),  # Just (x, y)
                nn.Sigmoid()  # Direct [0, 1] output
            ) for _ in range(self.num_cameras)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            [robot_state_feature] (optional): (B, state_dim) batch of robot states.

            [image_features]: (B, n_cameras, C, H, W) batch of images.
                AND/OR
            [env_state_feature]: (B, env_dim) batch of environment states.

            [action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
            
            [action] (optional, if use_action_chunk_input=True): (B, chunk_size, action_dim) batch of action sequences.
                Can be masked during co-training scenarios.
                
            [reward_horizons] (optional, for training): (B, len(horizon_steps)) batch of reward targets for each horizon step.
            
            Co-training mode (when use_cotraining=True):
            - Batch is automatically split based on cotraining_ratio
            - Action-focused samples: action is masked (zeroed), only action loss computed
            - Reward-focused samples: action provided as input, only reward loss computed
            - Gaze loss is computed for all samples regardless of focus
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            (B, 1, len(horizon_steps)) batch of reward predictions (if use_reward_head=True), else None
            List of gaze predictions for each camera
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        if self.config.use_vae and self.training:
            assert "action" in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        if "observation.images" in batch:
            batch_size = batch["observation.images"][0].shape[0]
        else:
            batch_size = batch["observation.environment_state"].shape[0]

        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and "action" in batch:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch["observation.state"])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch["action"])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch["observation.state"].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)

            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch["observation.state"].device
            )

        # Prepare transformer encoder inputs.
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        # Robot state token.
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch["observation.state"]))
        # Environment state token.
        if self.config.env_state_feature:
            encoder_in_tokens.append(
                self.encoder_env_state_input_proj(batch["observation.environment_state"])
            )
        
        # Action chunk tokens (if provided and not masked out)
        if self.config.use_action_chunk_input and "action" in batch:
            # Use action_for_encoder if available (co-training), otherwise use regular action
            action_input = batch.get("action_for_encoder", batch["action"])
            
            # Project action chunk: (B, chunk_size, action_dim) -> (B, chunk_size, dim_model)
            action_chunk_embed = self.encoder_action_chunk_input_proj(action_input)
            # Convert to list of tensors for each timestep: (chunk_size, B, dim_model)
            action_chunk_tokens = action_chunk_embed.transpose(0, 1)  # (chunk_size, B, dim_model)
            encoder_in_tokens.extend([action_chunk_tokens[i] for i in range(action_chunk_tokens.size(0))])
            # Add positional embeddings for action chunk
            action_chunk_pos_embed = self.encoder_action_chunk_pos_embed.weight.unsqueeze(1)  # (chunk_size, 1, dim_model)
            encoder_in_pos_embed.extend([action_chunk_pos_embed[i] for i in range(action_chunk_pos_embed.size(0))])

        # Camera observation features and positional embeddings.
        if self.config.image_features:
            all_cam_features = []
            all_cam_pos_embeds = []

            # For a list of images, the H and W may vary but H*W is constant.
            for img in batch["observation.images"]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # Rearrange features to (sequence, batch, dim).
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                all_cam_features.append(cam_features)
                all_cam_pos_embeds.append(cam_pos_embed)

            encoder_in_tokens.extend(torch.cat(all_cam_features, axis=0))
            encoder_in_pos_embed.extend(torch.cat(all_cam_pos_embeds, axis=0))

        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)
        
        # Reward predictions for multiple horizon steps
        if self.config.use_reward_head:
            # Predict rewards for current timestep (first position in chunk) across all horizon steps
            reward_preds = self.reward_head(decoder_out[:, 0:1, :])  # (B, 1, len(horizon_steps))
        else:
            reward_preds = None

        # Gaze predictions - one prediction per camera
        gaze_predictions = []
        for gaze_head in self.gaze_heads:
            gaze_pred = gaze_head(decoder_out)  # (B, chunk_size, 2)
            gaze_predictions.append(gaze_pred)

        return actions, reward_preds, gaze_predictions, (mu, log_sigma_x2)


class ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: REWACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: REWACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, config: REWACTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: REWACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            decoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            encoder_pos_embed: (DS, 1, C) Positional_embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
