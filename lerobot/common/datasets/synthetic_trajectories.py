#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Synthetic trajectory generation for data augmentation and ablation studies.

This module provides functionality to generate modified synthetic trajectories from existing data
for training robustness and conducting ablation studies, particularly useful for reward-based models.

Key Features:
- Reward Completion: Forces rewards to 1.0 after successful task completion
- Backwards Trajectories: Uses past observations as future actions (requires delta_timestamps)
- Stationary Trajectories: Freezes actions/rewards from a chosen temporal point
- Big Jumps: Adds dangerous action discontinuities with zero rewards

Note: Backwards trajectories require the policy configuration to include observation_delta_indices
and reward_delta_indices to provide access to past temporal data via the delta_timestamps mechanism.
"""

import logging
import random
from typing import Dict, Any, Optional, List
from enum import Enum

import numpy as np
import torch
from torch import Tensor


class SyntheticTrajectoryType(Enum):
    """Types of synthetic trajectories that can be generated."""
    REWARD_COMPLETION = "reward_completion"  # Force rewards to 1 after task completion
    BACKWARDS = "backwards"  # Use past states as future actions, past rewards as future rewards
    STATIONARY = "stationary"  # Keep actions and rewards constant from a chosen point
    BIG_JUMPS = "big_jumps"  # Add sudden action jumps with zero rewards


class SyntheticTrajectoryGenerator:
    """
    Generator for creating synthetic trajectory variants for data augmentation and ablation studies.
    
    This class provides methods to generate modified trajectories that can help with:
    - Testing model robustness to different reward patterns
    - Understanding the importance of temporal consistency
    - Evaluating safety behaviors (big jumps)
    - Studying reward completion dynamics
    """
    
    def __init__(
        self,
        synthetic_probability: float = 0.25,
        reward_completion_prob: float = 0.25,
        backwards_prob: float = 0.25,
        stationary_prob: float = 0.25,
        big_jumps_prob: float = 0.25,
        reward_completion_threshold: float = 0.9,
        reward_completion_history_length: int = 20,
        backwards_min_reward: float = 0.05,
        big_jump_min_delta: float = 0.4,
        big_jump_max_delta: float = 1.0,
        big_jump_max_joints: int = 3,
        seed: Optional[int] = None,
    ):
        """
        Initialize the synthetic trajectory generator.
        
        Args:
            synthetic_probability: Probability of generating a synthetic trajectory (0.0 to 1.0)
            reward_completion_prob: Probability of reward completion type (relative to other types)
            backwards_prob: Probability of backwards type (relative to other types)
            stationary_prob: Probability of stationary type (relative to other types)
            big_jumps_prob: Probability of big jumps type (relative to other types)
            reward_completion_threshold: Reward threshold to consider task "complete"
            reward_completion_history_length: How far back to check for reward completion
            backwards_min_reward: Minimum reward to maintain in backwards trajectories
            big_jump_min_delta: Minimum action delta for big jumps
            big_jump_max_delta: Maximum action delta for big jumps
            big_jump_max_joints: Maximum number of joints to modify in big jumps
            seed: Random seed for reproducibility
        """
        self.synthetic_probability = synthetic_probability
        
        # Normalize probabilities for trajectory types
        total_prob = reward_completion_prob + backwards_prob + stationary_prob + big_jumps_prob
        self.type_probabilities = {
            SyntheticTrajectoryType.REWARD_COMPLETION: reward_completion_prob / total_prob,
            SyntheticTrajectoryType.BACKWARDS: backwards_prob / total_prob,
            SyntheticTrajectoryType.STATIONARY: stationary_prob / total_prob,
            SyntheticTrajectoryType.BIG_JUMPS: big_jumps_prob / total_prob,
        }
        
        # Parameters for different trajectory types
        self.reward_completion_threshold = reward_completion_threshold
        self.reward_completion_history_length = reward_completion_history_length
        self.backwards_min_reward = backwards_min_reward
        self.big_jump_min_delta = big_jump_min_delta
        self.big_jump_max_delta = big_jump_max_delta
        self.big_jump_max_joints = big_jump_max_joints
        
        # Initialize random number generator
        self.rng = random.Random(seed) if seed is not None else random.Random()
        
        logging.info(f"SyntheticTrajectoryGenerator initialized with {synthetic_probability:.1%} probability")
        
    def should_generate_synthetic(self) -> bool:
        """Determine whether to generate a synthetic trajectory."""
        return self.rng.random() < self.synthetic_probability
    
    def choose_trajectory_type(self) -> SyntheticTrajectoryType:
        """Randomly choose which type of synthetic trajectory to generate."""
        rand_val = self.rng.random()
        cumulative_prob = 0.0
        
        for traj_type, prob in self.type_probabilities.items():
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return traj_type
        
        # Fallback (shouldn't happen with proper probabilities)
        return SyntheticTrajectoryType.REWARD_COMPLETION
    
    def generate_synthetic_trajectory(self, batch_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a synthetic trajectory from a real trajectory.
        
        Args:
            batch_item: Original trajectory data from dataset
            
        Returns:
            Modified trajectory data with synthetic modifications
        """
        # Always add metadata keys to ensure consistent batch structure
        if not self.should_generate_synthetic():
            # For non-synthetic items, add metadata indicating they are real
            batch_item['synthetic_trajectory_type'] = 'real'
            batch_item['is_synthetic'] = False
            return batch_item
        
        trajectory_type = self.choose_trajectory_type()
        
        # Create a copy to avoid modifying the original
        synthetic_item = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch_item.items()}
        
        # Apply the appropriate transformation
        if trajectory_type == SyntheticTrajectoryType.REWARD_COMPLETION:
            synthetic_item = self._generate_reward_completion(synthetic_item)
        elif trajectory_type == SyntheticTrajectoryType.BACKWARDS:
            synthetic_item = self._generate_backwards_trajectory(synthetic_item)
        elif trajectory_type == SyntheticTrajectoryType.STATIONARY:
            synthetic_item = self._generate_stationary_trajectory(synthetic_item)
        elif trajectory_type == SyntheticTrajectoryType.BIG_JUMPS:
            synthetic_item = self._generate_big_jumps_trajectory(synthetic_item)
        
        # Add metadata about the synthetic trajectory
        synthetic_item['synthetic_trajectory_type'] = trajectory_type.value
        synthetic_item['is_synthetic'] = True
        
        return synthetic_item
    
    def _generate_reward_completion(self, batch_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trajectory with reward completion pattern.
        
        After detecting high reward achievement, force all subsequent rewards to 1.0.
        """
        if 'reward' not in batch_item:
            return batch_item
        
        rewards = batch_item['reward']
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.clone()
        else:
            rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # Find if and when reward completion occurs
        completion_mask = rewards >= self.reward_completion_threshold
        
        if completion_mask.any():
            # Find the first completion point
            completion_indices = torch.where(completion_mask)[0]
            first_completion = completion_indices[0].item()
            
            # Check if completion is sustained for the history length
            history_start = max(0, first_completion - self.reward_completion_history_length)
            history_end = min(len(rewards), first_completion + self.reward_completion_history_length)
            
            # If there's sufficient evidence of completion, set all subsequent rewards to 1.0
            if (rewards[history_start:history_end] >= self.reward_completion_threshold).float().mean() > 0.5:
                rewards[first_completion:] = 1.0
                logging.debug(f"Applied reward completion from index {first_completion}")
        
        batch_item['reward'] = rewards
        return batch_item
    
    def _generate_backwards_trajectory(self, batch_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate backwards trajectory using past observations as future actions.
        
        This creates a trajectory where the robot "imagines" going backwards through
        previously visited states by using past observation.state as future actions.
        
        Note: This requires that the dataset was loaded with observation_delta_indices
        to provide access to past observations.
        """
        # Look for past observations in the batch (from delta_timestamps)
        past_obs_keys = [key for key in batch_item.keys() if key.startswith('observation.state') and '_is_pad' not in key]
        
        if not past_obs_keys:
            logging.warning("No past observations available for backwards trajectory generation")
            return batch_item
        
        # Find the key with the most temporal data (likely the main observation.state)
        main_obs_key = 'observation.state' if 'observation.state' in past_obs_keys else past_obs_keys[0]
        
        if main_obs_key not in batch_item:
            logging.warning(f"Observation key {main_obs_key} not found in batch")
            return batch_item
            
        past_observations = batch_item[main_obs_key]
        if isinstance(past_observations, torch.Tensor):
            past_observations = past_observations.clone()
        else:
            past_observations = torch.tensor(past_observations, dtype=torch.float32)
        
        # Handle different observation shapes
        if len(past_observations.shape) == 1:
            # Single timestep, can't create backwards trajectory
            logging.debug("Single timestep observation, cannot create backwards trajectory")
            return batch_item
        elif len(past_observations.shape) == 2:
            # (timesteps, obs_dim) - use past observations as future actions
            timesteps, obs_dim = past_observations.shape
            
            # Use these reversed past observations as our new action sequence
            if 'action' in batch_item:
                action_shape = batch_item['action'].shape
                if len(action_shape) == 2 and action_shape[1] <= obs_dim:
                    required_action_length = action_shape[0]  # Must match original action length
                    action_dim = action_shape[1]
                    
                    # Check if we have enough past observations
                    if timesteps < required_action_length:
                        logging.debug(f"Insufficient past observations ({timesteps}) for required action length ({required_action_length})")
                        return batch_item
                    
                    # Take exactly the required number of past observations and reverse them
                    # Use the most recent past observations (excluding current timestep)
                    start_idx = max(0, timesteps - required_action_length)
                    past_obs_segment = past_observations[start_idx:start_idx + required_action_length]
                    reversed_past_obs = torch.flip(past_obs_segment, dims=[0])  # Reverse temporal order
                    
                    # Project observation dimensions to action dimensions if needed
                    new_actions = reversed_past_obs[:, :action_dim]
                    batch_item['action'] = new_actions
                else:
                    logging.debug("Action shape incompatible with observation shape for backwards trajectory")
                    return batch_item
        
        # Handle rewards - simply flip the temporal reward sequence for backwards trajectories
        reward_keys = [key for key in batch_item.keys() if key == 'reward' or key.startswith('reward')]
        
        for reward_key in reward_keys:
            if reward_key in batch_item:
                rewards = batch_item[reward_key]
                if isinstance(rewards, torch.Tensor):
                    rewards = rewards.clone()
                else:
                    rewards = torch.tensor(rewards, dtype=torch.float32)
                
                # For backwards trajectories, simply flip the entire temporal reward sequence
                if len(rewards.shape) > 0 and rewards.shape[0] > 1:
                    flipped_rewards = torch.flip(rewards, dims=[0])
                    # Apply minimum reward threshold
                    flipped_rewards = torch.clamp(flipped_rewards, min=self.backwards_min_reward)
                    batch_item[reward_key] = flipped_rewards
        
        logging.debug("Applied backwards trajectory transformation using past observations")
        return batch_item
    
    def _generate_stationary_trajectory(self, batch_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate stationary trajectory by freezing actions and rewards from a chosen point.
        """
        if 'action' not in batch_item:
            return batch_item
        
        actions = batch_item['action']
        if isinstance(actions, torch.Tensor):
            actions = actions.clone()
        else:
            actions = torch.tensor(actions, dtype=torch.float32)
        
        # Choose a random stationary point (not too early in the sequence)
        sequence_length = len(actions) if len(actions.shape) == 1 else actions.shape[0]
        min_point = min(5, sequence_length // 4)  # Don't freeze too early
        max_point = sequence_length - 1
        
        if min_point < max_point:
            stationary_point = self.rng.randint(min_point, max_point)
            
            # Freeze actions from this point onwards
            if len(actions.shape) > 1:
                actions[stationary_point:] = actions[stationary_point:stationary_point+1].expand_as(actions[stationary_point:])
            else:
                actions[stationary_point:] = actions[stationary_point]
            
            batch_item['action'] = actions
            
            # Also freeze rewards if available
            if 'reward' in batch_item:
                rewards = batch_item['reward']
                if isinstance(rewards, torch.Tensor):
                    rewards = rewards.clone()
                else:
                    rewards = torch.tensor(rewards, dtype=torch.float32)
                
                rewards[stationary_point:] = rewards[stationary_point]
                batch_item['reward'] = rewards
            
            logging.debug(f"Applied stationary trajectory from index {stationary_point}")
        
        return batch_item
    
    def _generate_big_jumps_trajectory(self, batch_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trajectory with sudden big jumps in actions, setting subsequent rewards to 0.
        
        This simulates dangerous or erratic behavior that should be avoided.
        """
        if 'action' not in batch_item:
            return batch_item
        
        actions = batch_item['action']
        if isinstance(actions, torch.Tensor):
            actions = actions.clone()
        else:
            actions = torch.tensor(actions, dtype=torch.float32)
        
        # Determine action dimensions
        if len(actions.shape) == 1:
            sequence_length = len(actions)
            action_dim = 1
        else:
            sequence_length, action_dim = actions.shape[0], actions.shape[1]
        
        # Choose when to introduce the big jump (not too early)
        min_jump_point = min(10, sequence_length // 3)
        max_jump_point = sequence_length - 1
        
        if min_jump_point < max_jump_point:
            jump_point = self.rng.randint(min_jump_point, max_jump_point)
            
            # Choose which joints to modify (1 to max_joints)
            if action_dim > 1:
                num_joints_to_modify = self.rng.randint(1, min(self.big_jump_max_joints, action_dim))
                joint_indices = self.rng.sample(range(action_dim), num_joints_to_modify)
            else:
                joint_indices = [0]
            
            # Apply big jumps
            for joint_idx in joint_indices:
                # Random delta in the specified range, with random sign
                delta_magnitude = self.rng.uniform(self.big_jump_min_delta, self.big_jump_max_delta)
                delta_sign = self.rng.choice([-1, 1])
                delta = delta_magnitude * delta_sign
                
                if len(actions.shape) == 1:
                    actions[jump_point] += delta
                else:
                    actions[jump_point, joint_idx] += delta
            
            batch_item['action'] = actions
            
            # Set rewards to 0 from jump point onwards
            if 'reward' in batch_item:
                rewards = batch_item['reward']
                if isinstance(rewards, torch.Tensor):
                    rewards = rewards.clone()
                else:
                    rewards = torch.tensor(rewards, dtype=torch.float32)
                
                rewards[jump_point:] = 0.0
                batch_item['reward'] = rewards
            
            logging.debug(f"Applied big jump at index {jump_point} to joints {joint_indices}")
        
        return batch_item


def integrate_synthetic_trajectories_into_dataset():
    """
    Documentation function showing how to integrate synthetic trajectories into LeRobotDataset.
    
    This would be added to the LeRobotDataset.__init__ method:
    
    ```python
    def __init__(self, ..., use_synthetic_trajectories=False, synthetic_config=None):
        # ... existing initialization ...
        
        if use_synthetic_trajectories:
            synthetic_config = synthetic_config or {}
            self.synthetic_generator = SyntheticTrajectoryGenerator(**synthetic_config)
        else:
            self.synthetic_generator = None
    ```
    
    And in the __getitem__ method:
    
    ```python
    def __getitem__(self, idx):
        item = # ... existing item retrieval logic ...
        
        # Apply synthetic trajectory generation if enabled
        if self.synthetic_generator is not None:
            item = self.synthetic_generator.generate_synthetic_trajectory(item)
        
        return item
    ```
    """
    pass
