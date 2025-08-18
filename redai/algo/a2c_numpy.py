"""A2C (Actor-Critic) algorithm implementation with GAE."""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from ..nets.mlp_numpy import MLP


class ActorCritic:
    """
    Actor-Critic implementation with shared encoder and separate heads.
    
    Uses Generalized Advantage Estimation (GAE) for variance reduction
    and supports both discrete action spaces and value function learning.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_dim: int = 256,
        seed: int = 0
    ):
        """
        Initialize Actor-Critic networks.
        
        Args:
            input_dim: Dimension of input observations
            n_actions: Number of discrete actions
            hidden_dim: Hidden layer dimension
            seed: Random seed for initialization
        """
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        
        # Shared encoder network
        self.encoder = MLP(
            sizes=[input_dim, hidden_dim, hidden_dim],
            activation="relu",
            seed=seed
        )
        
        # Policy head (actor)
        self.policy_head = MLP(
            sizes=[hidden_dim, n_actions],
            activation="relu",
            output_activation="softmax",
            seed=seed + 1
        )
        
        # Value head (critic)
        self.value_head = MLP(
            sizes=[hidden_dim, 1],
            activation="relu",
            seed=seed + 2
        )
    
    def _encode(self, obs: np.ndarray) -> np.ndarray:
        """Encode observations through shared encoder."""
        return self.encoder.forward(obs)
    
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
        
        # Encode observation
        encoded = self._encode(obs)
        
        # Get policy logits and value
        policy_logits = self.policy_head.forward(encoded, store_cache=False)
        value = self.value_head.forward(encoded, store_cache=False)
        
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
            Dictionary with log_probs, values, and entropy
        """
        # Encode observations
        encoded = self._encode(obs_batch)
        
        # Get policy logits and values
        policy_logits = self.policy_head.forward(encoded, store_cache=False)
        values = self.value_head.forward(encoded, store_cache=False).squeeze(-1)
        
        # Convert logits to probabilities
        action_probs = policy_logits
        
        # Compute log probabilities for taken actions
        batch_size = obs_batch.shape[0]
        log_probs = np.log(action_probs[np.arange(batch_size), act_batch] + 1e-8)
        
        # Compute entropy
        entropy = self._compute_entropy(action_probs)
        
        return {
            'log_probs': log_probs,
            'values': values,
            'entropy': entropy,
            'action_probs': action_probs
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
        lr_policy: float,
        lr_value: float,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        grad_clip: float = 5.0
    ) -> Dict[str, float]:
        """
        Update actor-critic networks using policy gradients.
        
        Args:
            obs_batch: Batch of observations
            act_batch: Batch of actions
            advantages: Computed advantages
            returns: Computed returns
            lr_policy: Policy learning rate
            lr_value: Value learning rate
            entropy_coeff: Entropy regularization coefficient
            value_coeff: Value loss coefficient
            grad_clip: Gradient clipping threshold
            
        Returns:
            Dictionary with training statistics
        """
        batch_size = obs_batch.shape[0]
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Forward pass through networks
        encoded = self.encoder.forward(obs_batch, store_cache=True)
        policy_logits = self.policy_head.forward(encoded, store_cache=True)
        values = self.value_head.forward(encoded, store_cache=True).squeeze(-1)
        
        # Get action probabilities
        action_probs = policy_logits
        
        # Compute losses
        log_probs = np.log(action_probs[np.arange(batch_size), act_batch] + 1e-8)
        entropy = self._compute_entropy(action_probs)
        
        # Policy loss (negative because we want to maximize)
        policy_loss = -np.mean(log_probs * advantages)
        
        # Value loss
        value_loss = np.mean((values - returns) ** 2)
        
        # Total loss
        total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy
        
        # Backward pass for value head
        value_error = values - returns
        value_grad = 2 * value_error.reshape(-1, 1) / batch_size
        self.value_head.backward(value_grad)
        
        # Get gradients for encoder from value head
        # Fix: proper matrix multiplication for gradients
        # value_grad: (batch_size, 1), weights[0]: (hidden_dim, 1)
        # encoder gradient: (batch_size, 1) @ (1, hidden_dim) = (batch_size, hidden_dim)
        # Then sum over batch dimension: (hidden_dim,)
        value_encoder_grad = np.sum(np.dot(value_grad, self.value_head.weights[0].T), axis=0)
        
        # Backward pass for policy head
        policy_grad = np.zeros_like(action_probs)
        for i in range(batch_size):
            policy_grad[i, act_batch[i]] = -advantages[i] / (action_probs[i, act_batch[i]] + 1e-8)
        
        # Add entropy gradient
        entropy_grad = np.log(action_probs + 1e-8) + 1
        policy_grad += entropy_coeff * entropy_grad / batch_size
        
        policy_grad /= batch_size
        self.policy_head.backward(policy_grad)
        
        # Get gradients for encoder from policy head
        policy_encoder_grad = np.dot(policy_grad, self.policy_head.weights[0].T)
        
        # Combine encoder gradients
        total_encoder_grad = value_coeff * value_encoder_grad + policy_encoder_grad
        self.encoder.backward(total_encoder_grad)
        
        # Clip gradients
        policy_grad_norm = self.policy_head.clip_gradients(grad_clip)
        value_grad_norm = self.value_head.clip_gradients(grad_clip)
        encoder_grad_norm = self.encoder.clip_gradients(grad_clip)
        
        # Update networks
        self.encoder.step(lr_policy)
        self.policy_head.step(lr_policy)
        self.value_head.step(lr_value)
        
        # Clear gradients
        self.encoder.zero_grad()
        self.policy_head.zero_grad()
        self.value_head.zero_grad()
        
        return {
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'entropy': float(entropy),
            'total_loss': float(total_loss),
            'policy_grad_norm': float(policy_grad_norm),
            'value_grad_norm': float(value_grad_norm),
            'encoder_grad_norm': float(encoder_grad_norm),
            'advantage_mean': float(np.mean(advantages)),
            'advantage_std': float(np.std(advantages)),
            'return_mean': float(np.mean(returns)),
            'value_mean': float(np.mean(values))
        }


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    last_value: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation.
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        dones: List of done flags
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        last_value: Value estimate for last state
        
    Returns:
        Tuple of (advantages, returns)
    """
    advantages = []
    returns = []
    
    # Compute advantages using GAE
    gae = 0
    next_value = last_value
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
        
        next_value = values[t]
    
    return np.array(advantages), np.array(returns)