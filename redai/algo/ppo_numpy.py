"""PPO (Proximal Policy Optimization) algorithm implementation with clipping."""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from ..nets.mlp_numpy import MLP


class PPOActorCritic:
    """
    PPO implementation with clipped objective and KL divergence monitoring.
    
    Improves upon A2C with better sample efficiency and stability through
    clipped policy objectives and multiple epochs of updates per batch.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_dim: int = 512,  # Larger network for PPO
        seed: int = 0
    ):
        """
        Initialize PPO Actor-Critic networks.
        
        Args:
            input_dim: Dimension of input observations
            n_actions: Number of discrete actions
            hidden_dim: Hidden layer dimension
            seed: Random seed for initialization
        """
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        
        # Policy network (actor) - standalone 
        self.policy_net = MLP(
            sizes=[input_dim, hidden_dim, hidden_dim // 2, n_actions],
            activation="relu",
            output_activation="softmax",
            seed=seed
        )
        
        # Value network (critic) - standalone
        self.value_net = MLP(
            sizes=[input_dim, hidden_dim, hidden_dim // 2, 1],
            activation="relu",
            seed=seed + 1
        )
        
        # Store old policy for ratio computation
        self.old_policy_net = MLP(
            sizes=[input_dim, hidden_dim, hidden_dim // 2, n_actions],
            activation="relu",
            output_activation="softmax",
            seed=seed
        )
        
        # Copy initial policy parameters to old policy
        self._copy_policy_to_old()
    
    def _copy_policy_to_old(self):
        """Copy current policy parameters to old policy."""
        # Copy weights and biases from current policy to old policy
        for i in range(len(self.policy_net.weights)):
            self.old_policy_net.weights[i] = self.policy_net.weights[i].copy()
            self.old_policy_net.biases[i] = self.policy_net.biases[i].copy()
    
    # Remove shared encoder since we're using standalone networks
    
    def act(self, obs: np.ndarray) -> Tuple[int, float, float, Dict[str, Any]]:
        """
        Select action and compute value for given observation.
        
        Args:
            obs: Observation array
            
        Returns:
            Tuple of (action, log_prob, value, aux_info)
        """
        # Handle single observation
        single_obs = obs.ndim == 1
        if single_obs:
            obs = obs.reshape(1, -1)
        
        # Get policy logits and value directly from networks
        policy_logits = self.policy_net.forward(obs, store_cache=False)
        value = self.value_net.forward(obs, store_cache=False)
        
        # Convert logits to probabilities (softmax already applied in policy_head)
        action_probs = policy_logits
        
        # Sample action
        if single_obs:
            action_probs = action_probs.squeeze(0)
            value = value.squeeze(0)
            
            action = np.random.choice(self.n_actions, p=action_probs)
            log_prob = np.log(action_probs[action] + 1e-8)
        else:
            # Batch sampling
            actions = []
            log_probs = []
            
            for i in range(obs.shape[0]):
                action = np.random.choice(self.n_actions, p=action_probs[i])
                log_prob = np.log(action_probs[i, action] + 1e-8)
                actions.append(action)
                log_probs.append(log_prob)
            
            action = np.array(actions)
            log_prob = np.array(log_probs)
        
        aux_info = {
            'action_probs': action_probs,
            'entropy': self._compute_entropy(action_probs)
        }
        
        return action, log_prob, float(value) if single_obs else value, aux_info
    
    def evaluate(self, obs_batch: np.ndarray, act_batch: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Evaluate batch of observations and actions for policy updates.
        
        Args:
            obs_batch: Batch of observations
            act_batch: Batch of actions taken
            
        Returns:
            Dictionary with log_probs, values, entropy, and old_log_probs
        """
        # Get current policy logits and values directly
        policy_logits = self.policy_net.forward(obs_batch, store_cache=False)
        values = self.value_net.forward(obs_batch, store_cache=False).squeeze(-1)
        
        # Get old policy logits for ratio computation
        old_policy_logits = self.old_policy_net.forward(obs_batch, store_cache=False)
        
        # Convert logits to probabilities
        action_probs = policy_logits
        old_action_probs = old_policy_logits
        
        # Compute log probabilities for taken actions
        batch_size = obs_batch.shape[0]
        log_probs = np.log(action_probs[np.arange(batch_size), act_batch] + 1e-8)
        old_log_probs = np.log(old_action_probs[np.arange(batch_size), act_batch] + 1e-8)
        
        # Compute entropy
        entropy = self._compute_entropy(action_probs)
        
        # Compute KL divergence between old and new policy
        kl_div = np.mean(np.sum(old_action_probs * np.log(
            (old_action_probs + 1e-8) / (action_probs + 1e-8)), axis=-1))
        
        return {
            'log_probs': log_probs,
            'old_log_probs': old_log_probs,
            'values': values,
            'entropy': entropy,
            'action_probs': action_probs,
            'kl_divergence': kl_div
        }
    
    def _compute_entropy(self, action_probs: np.ndarray) -> float:
        """Compute policy entropy."""
        # Avoid log(0) by adding small epsilon
        log_probs = np.log(action_probs + 1e-8)
        entropy = -np.sum(action_probs * log_probs, axis=-1)
        return np.mean(entropy)
    
    def update(
        self,
        obs_batch: np.ndarray,
        act_batch: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        old_log_probs: np.ndarray,
        lr_policy: float,
        lr_value: float,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        grad_clip: float = 0.5,
        target_kl: float = 0.01
    ) -> Dict[str, float]:
        """
        Update PPO networks using clipped objective.
        
        Args:
            obs_batch: Batch of observations
            act_batch: Batch of actions
            advantages: Computed advantages
            returns: Computed returns
            old_log_probs: Log probabilities from old policy
            lr_policy: Policy learning rate
            lr_value: Value learning rate
            clip_ratio: PPO clipping parameter
            entropy_coeff: Entropy regularization coefficient
            value_coeff: Value loss coefficient
            grad_clip: Gradient clipping threshold
            target_kl: Target KL divergence for early stopping
            
        Returns:
            Dictionary with training statistics
        """
        batch_size = obs_batch.shape[0]
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Forward pass through standalone networks
        policy_logits = self.policy_net.forward(obs_batch, store_cache=True)
        values = self.value_net.forward(obs_batch, store_cache=True).squeeze(-1)
        
        # Get action probabilities (softmax already applied)
        action_probs = policy_logits
        
        # Compute current log probabilities
        log_probs = np.log(action_probs[np.arange(batch_size), act_batch] + 1e-8)
        
        # Compute probability ratio (new_prob / old_prob)
        ratio = np.exp(log_probs - old_log_probs)
        
        # Compute PPO clipped objective
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        policy_loss = -np.mean(np.minimum(surr1, surr2))
        
        # Compute value loss (MSE)
        value_loss = np.mean((values - returns) ** 2)
        
        # Compute entropy bonus
        entropy = self._compute_entropy(action_probs)
        entropy_loss = -entropy_coeff * entropy
        
        # Total loss
        total_loss = policy_loss + value_coeff * value_loss + entropy_loss
        
        # Backward pass - compute gradients
        # Value network gradients
        dloss_dvalue = 2.0 * value_coeff * (values - returns) / batch_size
        
        # Policy network gradients
        dloss_dlogits = np.zeros_like(action_probs)
        
        # PPO policy gradient
        for i in range(batch_size):
            action = act_batch[i]
            advantage = advantages[i]
            r = ratio[i]
            
            # Determine which term of min(surr1, surr2) is active
            if surr1[i] < surr2[i]:
                # surr1 is active: gradient from ratio * advantage
                grad_factor = -advantage / batch_size
            else:
                # surr2 is active: gradient from clipped ratio
                if r > 1.0 + clip_ratio or r < 1.0 - clip_ratio:
                    grad_factor = 0.0  # Gradient is zero due to clipping
                else:
                    grad_factor = -advantage / batch_size
            
            # Add gradient for the action taken (negative for loss minimization)
            dloss_dlogits[i, action] += grad_factor
        
        # Add entropy gradient (negative for loss minimization)
        dloss_dlogits -= entropy_coeff / batch_size * (np.log(action_probs + 1e-8) + 1.0)
        
        # Backward pass through networks
        self.policy_net.backward(dloss_dlogits)
        self.value_net.backward(dloss_dvalue.reshape(-1, 1))
        
        # Gradient clipping
        policy_grad_norm = self.policy_net.clip_gradients(grad_clip)
        value_grad_norm = self.value_net.clip_gradients(grad_clip)
        
        # Update parameters
        self.policy_net.step(lr_policy)
        self.value_net.step(lr_value)
        
        # Clear gradients
        self.policy_net.zero_grad()
        self.value_net.zero_grad()
        
        # Compute KL divergence for monitoring
        kl_div = np.mean(old_log_probs - log_probs + np.exp(log_probs - old_log_probs) - 1.0)
        
        # Compute explained variance for value function
        explained_var = 1 - np.var(returns - values) / (np.var(returns) + 1e-8)
        
        return {
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'entropy': float(entropy),
            'total_loss': float(total_loss),
            'kl_divergence': float(kl_div),
            'explained_variance': float(explained_var),
            'clip_fraction': float(np.mean(np.abs(ratio - 1.0) > clip_ratio)),
            'approx_kl': float(np.mean(log_probs - old_log_probs)),
            'ratio_mean': float(np.mean(ratio)),
            'advantages_mean': float(np.mean(advantages)),
            'advantages_std': float(np.std(advantages))
        }
    
    def _clip_gradients(self, grad_clip: float):
        """Clip gradients by global norm."""
        # Use the MLP's built-in gradient clipping for standalone networks
        policy_norm = self.policy_net.clip_gradients(grad_clip)
        value_norm = self.value_net.clip_gradients(grad_clip)
        return policy_norm + value_norm
    
    def update_old_policy(self):
        """Update old policy parameters with current policy (call after each epoch)."""
        self._copy_policy_to_old()
    
    def save_state(self) -> Dict[str, Any]:
        """Save network state for checkpointing."""
        return {
            'policy_net': self.policy_net.save_state(),
            'value_net': self.value_net.save_state(),
            'old_policy_net': self.old_policy_net.save_state()
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load network state from checkpoint."""
        self.policy_net.load_state(state['policy_net'])
        self.value_net.load_state(state['value_net'])
        self.old_policy_net.load_state(state['old_policy_net'])