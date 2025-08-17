"""Learned progress detector for identifying meaningful state transitions."""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from .mlp_numpy import MLP


class ProgressDetector:
    """
    Learns to identify progress in game states through temporal difference learning.
    
    Uses a neural network to predict value of states based on future exploration potential.
    Higher predicted values indicate states likely to lead to novel experiences.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        discount: float = 0.99,
        seed: int = 0
    ):
        """
        Initialize progress detector network.
        
        Args:
            input_dim: Dimension of input observations
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for value network
            discount: Discount factor for future rewards
            seed: Random seed
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.discount = discount
        
        # Value network predicts progress potential
        self.value_net = MLP(
            sizes=[input_dim, hidden_dim, hidden_dim, 1],
            activation="relu",
            seed=seed
        )
        
        # Difference network learns state transitions
        self.diff_net = MLP(
            sizes=[input_dim * 2, hidden_dim, hidden_dim // 2],
            activation="relu", 
            seed=seed + 1
        )
        
        # Statistics for normalization
        self.value_mean = 0.0
        self.value_var = 1.0
        self.transition_scores = []
        self.max_transition_score = 1e-8
        
    def compute_progress_reward(
        self, 
        obs: np.ndarray, 
        next_obs: np.ndarray,
        intrinsic_reward: float
    ) -> float:
        """
        Compute progress-based reward bonus.
        
        Args:
            obs: Current observation
            next_obs: Next observation
            intrinsic_reward: Base intrinsic reward from RND
            
        Returns:
            Combined progress and intrinsic reward
        """
        # Ensure observations are 2D
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        if next_obs.ndim == 1:
            next_obs = next_obs.reshape(1, -1)
            
        # Compute state values
        current_value = float(self.value_net.forward(obs, store_cache=False))
        next_value = float(self.value_net.forward(next_obs, store_cache=False))
        
        # Compute transition difference score
        transition_input = np.concatenate([obs, next_obs], axis=1)
        transition_score = float(self.diff_net.forward(transition_input, store_cache=False).mean())
        
        # Update max transition score for normalization
        self.max_transition_score = max(self.max_transition_score, abs(transition_score))
        
        # Temporal difference indicates progress
        td_progress = next_value - current_value
        
        # Normalized transition novelty
        normalized_transition = abs(transition_score) / (self.max_transition_score + 1e-8)
        
        # Combine signals: base intrinsic + learned progress + transition novelty
        progress_bonus = 0.3 * max(0, td_progress) + 0.2 * normalized_transition
        
        # Return weighted combination
        return intrinsic_reward * 0.7 + progress_bonus
    
    def update(
        self,
        obs_batch: np.ndarray,
        next_obs_batch: np.ndarray, 
        rewards_batch: np.ndarray
    ) -> Dict[str, float]:
        """
        Update networks based on observed transitions.
        
        Args:
            obs_batch: Batch of current observations
            next_obs_batch: Batch of next observations
            rewards_batch: Batch of intrinsic rewards
            
        Returns:
            Training statistics
        """
        if obs_batch.ndim == 1:
            obs_batch = obs_batch.reshape(1, -1)
        if next_obs_batch.ndim == 1:
            next_obs_batch = next_obs_batch.reshape(1, -1)
            
        batch_size = obs_batch.shape[0]
        
        # Forward pass for value network
        current_values = self.value_net.forward(obs_batch, store_cache=True)
        next_values = self.value_net.forward(next_obs_batch, store_cache=False)
        
        # Temporal difference target
        td_targets = rewards_batch.reshape(-1, 1) + self.discount * next_values
        
        # Value loss (MSE)
        value_error = td_targets - current_values
        value_loss = np.mean(value_error ** 2)
        
        # Backward pass for value network
        grad_output = -2 * value_error / batch_size
        self.value_net.backward(grad_output)
        
        # Gradient clipping and update
        value_grad_norm = self.value_net.clip_gradients(max_norm=5.0)
        self.value_net.step(self.learning_rate)
        self.value_net.zero_grad()
        
        # Update difference network
        transition_inputs = np.concatenate([obs_batch, next_obs_batch], axis=1)
        diff_outputs = self.diff_net.forward(transition_inputs, store_cache=True)
        
        # Difference loss - maximize variance in transitions
        diff_loss = -np.var(diff_outputs)
        
        # Backward pass for difference network
        diff_grad = 2 * (diff_outputs - np.mean(diff_outputs)) / batch_size
        self.diff_net.backward(diff_grad)
        
        # Update difference network
        diff_grad_norm = self.diff_net.clip_gradients(max_norm=5.0)
        self.diff_net.step(self.learning_rate * 0.5)  # Slower learning
        self.diff_net.zero_grad()
        
        # Update statistics
        self.value_mean = 0.99 * self.value_mean + 0.01 * np.mean(current_values)
        self.value_var = 0.99 * self.value_var + 0.01 * np.var(current_values)
        
        return {
            'progress_value_loss': float(value_loss),
            'progress_value_mean': float(self.value_mean),
            'progress_value_std': float(np.sqrt(self.value_var)),
            'value_grad_norm': float(value_grad_norm),
            'diff_grad_norm': float(diff_grad_norm),
            'max_transition_score': float(self.max_transition_score)
        }
    
    def compute_state_priority(self, obs: np.ndarray) -> float:
        """
        Compute priority score for a state (for archive scoring).
        
        Args:
            obs: Observation
            
        Returns:
            Priority score (higher = more promising for progress)
        """
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
            
        value = float(self.value_net.forward(obs, store_cache=False))
        
        # Normalize value to [0, 1] range
        normalized_value = (value - self.value_mean) / (np.sqrt(self.value_var) + 1e-8)
        return float(np.clip(normalized_value + 0.5, 0, 1))
    
    def save_state(self) -> Dict[str, Any]:
        """Save progress detector state."""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'discount': self.discount,
            'value_net': self.value_net.save_state(),
            'diff_net': self.diff_net.save_state(),
            'value_mean': self.value_mean,
            'value_var': self.value_var,
            'max_transition_score': self.max_transition_score
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load progress detector state."""
        self.input_dim = state['input_dim']
        self.hidden_dim = state['hidden_dim']
        self.learning_rate = state['learning_rate']
        self.discount = state['discount']
        
        # Recreate networks
        self.value_net = MLP([self.input_dim, self.hidden_dim, self.hidden_dim, 1], "relu")
        self.diff_net = MLP([self.input_dim * 2, self.hidden_dim, self.hidden_dim // 2], "relu")
        
        # Load network states
        self.value_net.load_state(state['value_net'])
        self.diff_net.load_state(state['diff_net'])
        
        # Load statistics
        self.value_mean = state['value_mean']
        self.value_var = state['value_var']
        self.max_transition_score = state['max_transition_score']