"""Random Network Distillation for curiosity-driven exploration."""

import numpy as np
from typing import Dict, Any, Optional
from .mlp_numpy import MLP


class RND:
    """
    Random Network Distillation implementation for intrinsic motivation.
    
    Uses a fixed random target network and a trainable predictor network.
    The prediction error serves as an intrinsic reward signal that encourages
    exploration of novel states.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        reward_clip: float = 5.0,
        norm_ema: float = 0.99,
        seed: int = 0,
        weight_decay: float = 1e-5,
        min_reward_std: float = 1e-4
    ):
        """
        Initialize RND networks and normalization.
        
        Args:
            input_dim: Dimension of input observations
            hidden_dim: Hidden layer dimension for both networks
            reward_clip: Maximum intrinsic reward value
            norm_ema: EMA decay rate for reward normalization
            seed: Random seed for network initialization
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.reward_clip = reward_clip
        self.norm_ema = norm_ema
        self.weight_decay = weight_decay
        self.min_reward_std = min_reward_std
        
        # Target network (fixed, random weights)
        self.target_net = MLP(
            sizes=[input_dim, hidden_dim, hidden_dim],
            activation="relu",
            seed=seed
        )
        
        # Predictor network (trainable)
        self.predictor_net = MLP(
            sizes=[input_dim, hidden_dim, hidden_dim],
            activation="relu",
            seed=seed + 1  # Different initialization
        )
        
        # Reward normalization statistics
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0
        
        # Fallback exploration bonus
        self.fallback_bonus = 0.01
        
        # Reset tracking
        self.last_reset_step = 0
        self.reset_interval = 500000  # Reset predictor every 500K steps (much less aggressive)
        
        # Initialize with small batch to get reasonable initial statistics
        self._initialize_normalization()
    
    def _initialize_normalization(self) -> None:
        """Initialize reward normalization with random samples."""
        dummy_obs = np.random.randn(100, self.input_dim)
        dummy_rewards = []
        
        for i in range(100):
            reward = self._compute_raw_reward(dummy_obs[i])
            dummy_rewards.append(reward)
        
        if dummy_rewards:
            self.reward_mean = np.mean(dummy_rewards)
            self.reward_var = np.var(dummy_rewards) + 1e-8
    
    def _compute_raw_reward(self, obs: np.ndarray) -> float:
        """
        Compute raw intrinsic reward without normalization.
        
        Args:
            obs: Observation array
            
        Returns:
            Raw prediction error as intrinsic reward
        """
        # Ensure observation is 2D for network input
        if obs.ndim == 1:
            obs_batch = obs.reshape(1, -1)
        else:
            obs_batch = obs
        
        # Get target and prediction
        target_output = self.target_net.forward(obs_batch, store_cache=False)
        predictor_output = self.predictor_net.forward(obs_batch, store_cache=False)
        
        # Compute prediction error (MSE)
        error = np.mean((target_output - predictor_output) ** 2)
        return float(error)
    
    def intrinsic_reward(self, obs: np.ndarray) -> float:
        """
        Compute normalized intrinsic reward for a single observation.
        
        Args:
            obs: Observation array
            
        Returns:
            Normalized and clipped intrinsic reward
        """
        raw_reward = self._compute_raw_reward(obs)
        
        # Compute normalized reward with minimum std threshold
        reward_std = np.sqrt(self.reward_var + 1e-8)
        reward_std = max(reward_std, self.min_reward_std)  # Prevent collapse to zero
        
        normalized_reward = (raw_reward - self.reward_mean) / reward_std
        
        # Clip reward and add fallback exploration bonus
        clipped_reward = np.clip(normalized_reward, 0, self.reward_clip) + self.fallback_bonus
        
        # Update running statistics
        self._update_reward_stats(raw_reward)
        
        return float(clipped_reward)
    
    def _update_reward_stats(self, reward: float) -> None:
        """Update running reward statistics using EMA."""
        self.reward_count += 1
        
        # Update mean
        self.reward_mean = self.norm_ema * self.reward_mean + (1 - self.norm_ema) * reward
        
        # Update variance (using Welford's online algorithm concept with EMA)
        diff = reward - self.reward_mean
        self.reward_var = self.norm_ema * self.reward_var + (1 - self.norm_ema) * (diff ** 2)
    
    def update(self, obs_batch: np.ndarray, lr: float, global_step: int = 0) -> Dict[str, float]:
        """
        Update predictor network to minimize prediction error.
        
        Args:
            obs_batch: Batch of observations
            lr: Learning rate
            global_step: Global training step for reset tracking
            
        Returns:
            Dictionary with training statistics
        """
        if obs_batch.ndim == 1:
            obs_batch = obs_batch.reshape(1, -1)
        
        batch_size = obs_batch.shape[0]
        
        # Check if we need to reset predictor to prevent collapse
        # Make reset interval much longer and add safety checks
        if (global_step > 0 and 
            (global_step - self.last_reset_step) >= self.reset_interval and
            global_step > 200000):  # Only reset after substantial training
            try:
                print(f"RND: Attempting predictor reset at step {global_step}")
                self.reset_predictor()
                self.last_reset_step = global_step
                print(f"RND: Predictor reset successful")
            except Exception as e:
                print(f"RND: Reset failed, continuing with current predictor: {e}")
                self.last_reset_step = global_step  # Prevent retry immediately
        
        # Forward pass through both networks
        target_output = self.target_net.forward(obs_batch, store_cache=False)
        predictor_output = self.predictor_net.forward(obs_batch, store_cache=True)
        
        # Compute loss (MSE between target and predictor)
        prediction_error = target_output - predictor_output
        mse_loss = np.mean(prediction_error ** 2)
        
        # Add L2 regularization to prevent overfitting
        l2_loss = 0.0
        for layer in self.predictor_net.layers:
            if hasattr(layer, 'W'):
                l2_loss += np.sum(layer.W ** 2)
        
        total_loss = mse_loss + self.weight_decay * l2_loss
        
        # Backward pass for predictor network
        grad_output = -2 * prediction_error / batch_size
        self.predictor_net.backward(grad_output)
        
        # Add L2 regularization gradients
        if self.weight_decay > 0:
            for layer in self.predictor_net.layers:
                if hasattr(layer, 'dW'):
                    layer.dW += 2 * self.weight_decay * layer.W
        
        # Gradient clipping
        grad_norm = self.predictor_net.clip_gradients(max_norm=5.0)
        
        # Update predictor network
        self.predictor_net.step(lr)
        self.predictor_net.zero_grad()
        
        return {
            'rnd_loss': float(mse_loss),
            'rnd_total_loss': float(total_loss),
            'rnd_grad_norm': float(grad_norm),
            'reward_mean': float(self.reward_mean),
            'reward_std': float(np.sqrt(self.reward_var))
        }
    
    def batch_intrinsic_rewards(self, obs_batch: np.ndarray) -> np.ndarray:
        """
        Compute intrinsic rewards for a batch of observations.
        
        Args:
            obs_batch: Batch of observations
            
        Returns:
            Array of intrinsic rewards
        """
        if obs_batch.ndim == 1:
            obs_batch = obs_batch.reshape(1, -1)
        
        rewards = []
        for i in range(obs_batch.shape[0]):
            reward = self.intrinsic_reward(obs_batch[i])
            rewards.append(reward)
        
        return np.array(rewards)
    
    def reset_predictor(self) -> None:
        """
        Reset predictor network to prevent collapse.
        
        This reinitializes the predictor network weights to restore
        prediction errors and curiosity signal.
        """
        # Reinitialize predictor network with different seed
        new_seed = int(np.random.rand() * 10000)
        self.predictor_net = MLP(
            sizes=[self.input_dim, self.hidden_dim, self.hidden_dim],
            activation="relu",
            seed=new_seed
        )
        
        # Reset normalization statistics partially to prevent shock
        self.reward_var = max(self.reward_var, 1.0)  # Ensure minimum variance
        
        print(f"RND predictor network reset (seed={new_seed}) to restore curiosity")
    
    def normalize_reward(self, raw_reward: float) -> float:
        """
        Normalize a raw reward using current statistics.
        
        Args:
            raw_reward: Raw prediction error
            
        Returns:
            Normalized and clipped reward
        """
        normalized = (raw_reward - self.reward_mean) / np.sqrt(self.reward_var + 1e-8)
        return float(np.clip(normalized, 0, self.reward_clip))
    
    def get_stats(self) -> Dict[str, float]:
        """Get current RND statistics."""
        return {
            'reward_mean': float(self.reward_mean),
            'reward_std': float(np.sqrt(self.reward_var)),
            'reward_count': int(self.reward_count)
        }
    
    def save_state(self) -> Dict[str, Any]:
        """Save RND state for checkpointing."""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'reward_clip': self.reward_clip,
            'norm_ema': self.norm_ema,
            'target_net': self.target_net.save_state(),
            'predictor_net': self.predictor_net.save_state(),
            'reward_mean': self.reward_mean,
            'reward_var': self.reward_var,
            'reward_count': self.reward_count
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load RND state from checkpoint."""
        self.input_dim = state['input_dim']
        self.hidden_dim = state['hidden_dim']
        self.reward_clip = state['reward_clip']
        self.norm_ema = state['norm_ema']
        
        # Recreate networks
        self.target_net = MLP([self.input_dim, self.hidden_dim, self.hidden_dim], "relu")
        self.predictor_net = MLP([self.input_dim, self.hidden_dim, self.hidden_dim], "relu")
        
        # Load network states
        self.target_net.load_state(state['target_net'])
        self.predictor_net.load_state(state['predictor_net'])
        
        # Load normalization statistics
        self.reward_mean = state['reward_mean']
        self.reward_var = state['reward_var']
        self.reward_count = state['reward_count']