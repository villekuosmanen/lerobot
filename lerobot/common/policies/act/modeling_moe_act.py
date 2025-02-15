import math
from collections import deque
from itertools import chain
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import (
    ACTEncoder,
    ACTDecoder,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
    create_sinusoidal_pos_embedding,
)
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy


class MoEACTPolicy(PreTrainedPolicy):

    config_class = ACTConfig
    name = "act"

    """Policy wrapper that includes normalization and temporal ensembling."""
    def __init__(self, config: ACTConfig, dataset_stats: dict[str, dict[str, Tensor]] | None = None):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # TODO: remove this as it is quite hacky
        print(f"n_action_steps default: {self.config.n_action_steps}")
        # self.config.n_action_steps = 8
        # self.config.temporal_ensemble_coeff = 0.01
        print(f"n_action_steps override: {self.config.n_action_steps}")
        
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)
        
        # MoE model instead of original ACT
        self.model = MoEACT(config)
        
        # Keep temporal ensembling from original
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
            self._eoe_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )

        # If we are doing temporal ensembling, do online updates where we keep track of the number of actions
        # we are ensembling over.
        if self.config.temporal_ensemble_coeff is not None:
            actions, eoe_preds, _ = self.model(batch)
            actions = self.unnormalize_outputs({"action": actions})["action"]
            action = self.temporal_ensembler.update(actions)
            eoe_pred = eoe_preds[0, -1]  # Take last prediction
            return action, eoe_pred

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions, eoe_preds, _ = self.model(batch)
            actions = actions[:, :self.config.n_action_steps]
            eoe_preds = eoe_preds[:, :self.config.n_action_steps]


            # TODO(rcadene): make _forward return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
            self._eoe_queue.extend(eoe_preds.transpose(0, 1))
        return self._action_queue.popleft(), self._eoe_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Modified forward to include MoE-specific losses and VAE loss."""
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        batch = self.normalize_targets(batch)
        
        # Get model outputs including auxiliary info
        actions_hat, eoe_hat, aux_info = self.model(batch)
        
        # Original losses
        l1_loss = (
            F.l1_loss(batch["action"], actions_hat, reduction="none") 
            * ~batch["action_is_pad"].bool().unsqueeze(-1)
        ).mean()
        
        eoe_targets = batch["next.done"].unsqueeze(1).expand(-1, eoe_hat.size(1))
        eoe_loss = F.binary_cross_entropy_with_logits(
            eoe_hat.squeeze(-1),
            eoe_targets.float(),
            weight=(~batch["action_is_pad"].bool()).float()
        )

        # Compute KL divergence loss if using VAE
        kld_loss = None
        if self.config.use_vae:
            mu, log_sigma_x2 = aux_info['vae_params']
            if mu is not None:  # Check we got VAE parameters
                # KL divergence between learned distribution and standard normal
                # Formula: -0.5 * sum(1 + log(σ²) - μ² - σ²)
                kld_loss = (-0.5 * (
                    1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp()
                )).sum(-1).mean()

        # Combine all losses
        loss_dict = {
            "l1_loss": l1_loss.item(),
            "eoe_loss": eoe_loss.item(),
            "aux_loss": aux_info['aux_loss'].item(),
        }
        
        # Add KLD loss if computed
        if kld_loss is not None:
            loss_dict["kld_loss"] = kld_loss.item()
            loss_dict["loss"] = (
                l1_loss + 
                eoe_loss + 
                aux_info['aux_loss'] + 
                kld_loss * self.config.kl_weight
            )
        else:
            loss_dict["loss"] = l1_loss + eoe_loss + aux_info['aux_loss']
        
        # Add auxiliary metrics
        loss_dict.update(aux_info['metrics'])
        
        return loss_dict

class MoEACT(nn.Module):
    """MoE version of ACT that combines router and experts."""
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config

        # Shared VAE components
        if config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    config.robot_state_feature.shape[0], config.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(
                config.action_feature.shape[0], config.dim_model
            )
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            
            # VAE positional encoding
            num_input_token_encoder = 1 + config.chunk_size
            if config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )
            
        # Shared vision backbone
        if config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        
        # Router
        self.router = ACTRouter(config)
        
        # Experts
        self.experts = nn.ModuleList([ACTExpert(config) for _ in range(config.num_experts)])
        
        # Load balancing parameters
        self.capacity_factor = 1.0  # Can be tuned to allow experts to handle more/fewer tokens
        self.min_expert_capacity = 4  # Minimum number of samples per expert

        # Loss scaling coefficients
        self.load_balance_coeff = 0.01  # Start small, can be tuned
        self.overflow_coeff = 0.01      # Separate coefficient for over

    def _compute_vae_latent(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Compute VAE latent vector using shared VAE encoder.
        
        Returns:
            latent_sample: (batch_size, latent_dim) sampled latent vector
            (mu, log_sigma_x2): Distribution parameters for computing KL loss
        """
        batch_size = (
            batch["observation.images"].shape[0] 
            if "observation.images" in batch 
            else batch["observation.environment_state"].shape[0]
        )
        
        # Prepare VAE encoder input
        cls_embed = einops.repeat(
            self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
        )
        
        if self.config.robot_state_feature:
            robot_state_embed = self.vae_encoder_robot_state_input_proj(batch["observation.state"])
            robot_state_embed = robot_state_embed.unsqueeze(1)
            
        action_embed = self.vae_encoder_action_input_proj(batch["action"])
        
        # Combine embeddings
        if self.config.robot_state_feature:
            vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
        else:
            vae_encoder_input = [cls_embed, action_embed]
        vae_encoder_input = torch.cat(vae_encoder_input, axis=1)
        
        # Get positional embedding
        pos_embed = self.vae_encoder_pos_enc.clone().detach()
        
        # Prepare padding mask
        cls_joint_is_pad = torch.full(
            (batch_size, 2 if self.config.robot_state_feature else 1),
            False,
            device=batch["observation.state"].device,
        )
        key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)
        
        # Get latent distribution parameters
        cls_token_out = self.vae_encoder(
            vae_encoder_input.permute(1, 0, 2),
            pos_embed=pos_embed.permute(1, 0, 2),
            key_padding_mask=key_padding_mask,
        )[0]
        
        latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
        mu = latent_pdf_params[:, :self.config.latent_dim]
        log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim:]
        
        # Sample latent using reparameterization trick
        latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        
        return latent_sample, (mu, log_sigma_x2)

    def _compute_expert_capacity(self, batch_size: int) -> int:
        """Compute how many samples each expert should handle."""
        capacity = int(self.capacity_factor * batch_size * self.config.top_k / self.config.num_experts)
        return max(capacity, self.min_expert_capacity)
    
    def _compute_load_balancing_loss(self, routing_probs: Tensor, expert_mask: Tensor) -> tuple[Tensor, dict]:
        """Compute load balancing auxiliary losses with proper scaling."""
        # Compute fraction of tokens routed to each expert
        expert_usage = expert_mask.float().mean(dim=0)
        
        # Compute load balancing loss (all experts should be used equally)
        target_usage = torch.ones_like(expert_usage) / self.config.num_experts
        load_balance_loss = F.mse_loss(expert_usage, target_usage)
        
        # Compute overflow loss
        routing_mass = routing_probs.sum(dim=0)
        overflow_loss = torch.max(
            routing_mass - torch.ones_like(routing_mass) * self.capacity_factor,
            torch.zeros_like(routing_mass)
        ).mean()
        
        # Scale losses separately
        scaled_load_balance_loss = self.load_balance_coeff * load_balance_loss
        scaled_overflow_loss = self.overflow_coeff * overflow_loss
        
        # Compute metrics including unscaled losses for monitoring
        metrics = {
            'unscaled_load_balance_loss': load_balance_loss.item(),
            'unscaled_overflow_loss': overflow_loss.item(),
            'scaled_load_balance_loss': scaled_load_balance_loss.item(),
            'scaled_overflow_loss': scaled_overflow_loss.item()
        }
        
        # Add expert usage statistics
        for i, usage in enumerate(expert_usage.cpu().numpy()):
            metrics[f'expert_{i}_usage'] = usage.item()
        
        # Track the imbalance
        max_usage = expert_usage.max().item()
        min_usage = expert_usage.min().item()
        metrics['expert_usage_imbalance'] = max_usage - min_usage
            
        total_aux_loss = scaled_load_balance_loss + scaled_overflow_loss
        return total_aux_loss, metrics
    
    def _process_vision_features(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]] | None:
        """Process vision features using shared backbone."""
        if not self.config.image_features:
            return None
            
        all_cam_features = []
        all_cam_pos_embeds = []
        
        for cam_index in range(batch["observation.images"].shape[-4]):
            cam_features = self.backbone(batch["observation.images"][:, cam_index])["feature_map"]
            cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
            cam_features = self.encoder_img_feat_input_proj(cam_features)
            all_cam_features.append(cam_features)
            all_cam_pos_embeds.append(cam_pos_embed)
            
        # Process for transformer input
        all_cam_features = torch.cat(all_cam_features, axis=-1)
        features = einops.rearrange(all_cam_features, "b c h w -> (h w) b c")
        
        all_cam_pos_embeds = torch.cat(all_cam_pos_embeds, axis=-1)
        pos_embeds = einops.rearrange(all_cam_pos_embeds, "b c h w -> (h w) b c")
        
        return features, pos_embeds

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, dict]:
        """
        Returns:
            combined_actions: Combined expert outputs
            combined_eoe: Combined end-of-episode predictions
            aux_info: Dictionary containing routing decisions, losses and metrics
        """
        batch_size = batch["observation.images"].shape[0] if "observation.images" in batch else batch["observation.environment_state"].shape[0]
        
        if self.config.use_vae and "action" in batch:
            latent_sample, (mu, log_sigma_x2) = self._compute_vae_latent(batch)
        else:
            latent_sample = torch.zeros(
                [batch_size, self.config.latent_dim], 
                dtype=torch.float32,
                device=batch["observation.state"].device
            )
            mu = log_sigma_x2 = None
        # Process vision features if needed
        processed_image_features = self._process_vision_features(batch) if self.config.image_features else None
        
        
        # Get routing decisions
        routing_logits = self.router(batch)
        routing_probs = F.softmax(routing_logits, dim=-1)
        
        # Get top k experts
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.config.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # Renormalize
        
        # Create expert assignment mask
        expert_mask = torch.zeros_like(routing_probs)
        expert_mask.scatter_(1, top_k_indices, 1)
        
        # Compute expert capacity
        expert_capacity = self._compute_expert_capacity(batch_size)
        
        # Initialize output tensors
        combined_actions = torch.zeros(
            (batch_size, self.config.chunk_size, self.config.action_feature.shape[0]),
            device=routing_logits.device
        )
        combined_eoe = torch.zeros(
            (batch_size, self.config.chunk_size, 1),
            device=routing_logits.device
        )
        
        # Track which experts were used for each sample
        expert_counts = torch.zeros(self.config.num_experts, device=routing_logits.device)
        
        # Compute weighted combination of expert outputs
        for i in range(self.config.top_k):
            expert_indices = top_k_indices[:, i]
            weights = top_k_probs[:, i].unsqueeze(-1).unsqueeze(-1)
            
            for j, expert_idx in enumerate(expert_indices):
                if expert_counts[expert_idx] > expert_capacity:
                    continue
                    
                actions, eoe = self.experts[expert_idx](
                    batch, 
                    latent_sample=latent_sample,
                    processed_image_features=processed_image_features
                )
                combined_actions[j] += weights[j] * actions[j]
                combined_eoe[j] += weights[j] * eoe[j]
        
        # Compute load balancing losses and metrics
        aux_loss, aux_metrics = self._compute_load_balancing_loss(routing_probs, expert_mask)
        
        aux_info = {
            'aux_loss': aux_loss,
            'routing_probs': routing_probs,
            'expert_mask': expert_mask,
            'metrics': aux_metrics
        }
        
        # Include VAE parameters in aux_info
        aux_info['vae_params'] = (mu, log_sigma_x2)
        return combined_actions, combined_eoe, aux_info

class ACTRouter(nn.Module):
    """Router module for MoE ACT."""
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config  # Need to store config for feature access
        
        # Input projection similar to original ACT encoder input
        if config.robot_state_feature:
            self.robot_state_proj = nn.Linear(config.robot_state_feature.shape[0], config.dim_model)
        if config.env_state_feature:
            self.env_state_proj = nn.Linear(config.env_state_feature.shape[0], config.dim_model)
        if config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
            self.img_feat_proj = nn.Conv2d(backbone_model.fc.in_features, config.dim_model, kernel_size=1)
        
        # Router network
        router_input_dim = config.dim_model
        if config.robot_state_feature:
            router_input_dim += config.dim_model
        if config.env_state_feature:
            router_input_dim += config.dim_model
        if config.image_features:
            router_input_dim += config.dim_model
            
        self.router_net = nn.Sequential(
            nn.Linear(router_input_dim, config.dim_model),
            nn.ReLU(),
            nn.Linear(config.dim_model, config.num_experts)
        )
        
    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Returns:
            routing_logits: (batch_size, num_experts) tensor of logits
        """
        features = []
        
        # Process each input type similar to original ACT
        if self.config.robot_state_feature:
            robot_state = self.robot_state_proj(batch["observation.state"])
            features.append(robot_state)
            
        if self.config.env_state_feature:
            env_state = self.env_state_proj(batch["observation.environment_state"])
            features.append(env_state)
            
        if self.config.image_features:
            # Average pool image features across spatial dimensions
            img_feats = []
            for cam_idx in range(batch["observation.images"].shape[-4]):
                cam_features = self.backbone(batch["observation.images"][:, cam_idx])["feature_map"]
                cam_features = self.img_feat_proj(cam_features)
                img_feats.append(cam_features.mean(dim=[-2, -1]))
            img_feats = torch.cat(img_feats, dim=1)
            features.append(img_feats)
            
        # Combine features and compute routing logits
        combined_features = torch.cat(features, dim=-1)
        routing_logits = self.router_net(combined_features)
        
        return routing_logits  # Just return the logits, let MoEACT handle the softmax

class ACTExpert(nn.Module):
    """Lightweight expert that contains only the necessary components for prediction."""
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config
        
        # Core transformer components
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)
        
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)

        # Input projections for transformer
        if config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.robot_state_feature.shape[0], config.dim_model
            )
        if config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                config.env_state_feature.shape[0], config.dim_model
            )
        
        # Transformer encoder positional embeddings
        n_1d_tokens = 1  # for the latent
        if config.robot_state_feature:
            n_1d_tokens += 1
        if config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        
        # Decoder position embedding
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        
        # Output heads
        self.action_head = nn.Linear(config.dim_model, config.action_feature.shape[0])
        self.eoe_head = nn.Linear(config.dim_model, 1)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, 
                batch: dict[str, Tensor],
                latent_sample: Tensor,
                processed_image_features: tuple[Tensor, Tensor] | None = None
               ) -> tuple[Tensor, Tensor]:
        """Forward pass for a single expert.
        
        Args:
            batch: Input batch dictionary
            latent_sample: (B, latent_dim) Latent vector from VAE
            processed_image_features: Optional tuple of (features, pos_embeds) from shared vision backbone
        """
        batch_size = (
            batch["observation.images"]
            if "observation.images" in batch
            else batch["observation.environment_state"]
        ).shape[0]

        # Prepare transformer encoder inputs
        projected_latent = self.encoder_latent_input_proj(latent_sample)
        encoder_in_tokens = [projected_latent]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        
        # Add robot and env state if present
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch["observation.state"]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(
                self.encoder_env_state_input_proj(batch["observation.environment_state"])
            )
            
        # Add processed image features if provided
        if processed_image_features is not None:
            features, pos_embeds = processed_image_features
            encoder_in_tokens.extend(features)
            encoder_in_pos_embed.extend(pos_embeds)
            
        # Stack tokens and pos embeddings
        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)
        
        # Forward through transformer
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
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
        
        # Generate outputs
        decoder_out = decoder_out.transpose(0, 1)
        actions = self.action_head(decoder_out)
        eoe_logits = self.eoe_head(decoder_out)
        
        return actions, eoe_logits
