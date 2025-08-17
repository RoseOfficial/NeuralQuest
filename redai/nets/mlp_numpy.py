"""NumPy-only Multi-Layer Perceptron implementation with backpropagation."""

import numpy as np
from typing import List, Optional, Dict, Any, Literal


class MLP:
    """
    Multi-Layer Perceptron implemented in pure NumPy.
    
    Supports custom architectures with configurable activation functions,
    gradient computation, and Adam optimization. Designed for RL applications
    where PyTorch/TensorFlow dependencies are not desired.
    """
    
    def __init__(
        self,
        sizes: List[int],
        activation: Literal["relu", "tanh", "sigmoid"] = "relu",
        output_activation: Optional[str] = None,
        seed: int = 0
    ):
        """
        Initialize MLP with specified architecture.
        
        Args:
            sizes: List of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
            activation: Activation function for hidden layers
            output_activation: Optional activation for output layer
            seed: Random seed for weight initialization
        """
        self.sizes = sizes
        self.activation = activation
        self.output_activation = output_activation
        self.num_layers = len(sizes) - 1
        
        np.random.seed(seed)
        
        # Initialize weights and biases using He initialization
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers):
            # He initialization for ReLU, Xavier for tanh/sigmoid
            if activation == "relu":
                std = np.sqrt(2.0 / sizes[i])
            else:
                std = np.sqrt(1.0 / sizes[i])
            
            w = np.random.normal(0, std, (sizes[i], sizes[i + 1]))
            b = np.zeros(sizes[i + 1])
            
            self.weights.append(w)
            self.biases.append(b)
        
        # Initialize Adam optimizer state
        self._reset_optimizer()
        
        # Cache for backpropagation
        self._activations = []
        self._z_values = []
        self._input = None
    
    def _reset_optimizer(self) -> None:
        """Reset Adam optimizer state."""
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0  # Time step for Adam
    
    def _activation_fn(self, x: np.ndarray, fn_type: str) -> np.ndarray:
        """Apply activation function."""
        if fn_type == "relu":
            return np.maximum(0, x)
        elif fn_type == "tanh":
            return np.tanh(x)
        elif fn_type == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
        elif fn_type == "softmax":
            # Stable softmax
            x_shifted = x - np.max(x, axis=-1, keepdims=True)
            exp_x = np.exp(x_shifted)
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        else:
            return x  # Linear/identity
    
    def _activation_derivative(self, x: np.ndarray, fn_type: str) -> np.ndarray:
        """Compute derivative of activation function."""
        if fn_type == "relu":
            return (x > 0).astype(np.float32)
        elif fn_type == "tanh":
            return 1 - np.tanh(x) ** 2
        elif fn_type == "sigmoid":
            sig = self._activation_fn(x, "sigmoid")
            return sig * (1 - sig)
        else:
            return np.ones_like(x)  # Linear
    
    def forward(self, x: np.ndarray, store_cache: bool = True) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            x: Input array of shape (batch_size, input_dim) or (input_dim,)
            store_cache: Whether to store intermediate values for backpropagation
            
        Returns:
            Output array of shape (batch_size, output_dim) or (output_dim,)
        """
        # Handle single sample input
        single_sample = x.ndim == 1
        if single_sample:
            x = x.reshape(1, -1)
        
        if store_cache:
            self._input = x.copy()
            self._activations = [x]
            self._z_values = []
        
        current = x
        
        for i in range(self.num_layers):
            # Linear transformation
            z = np.dot(current, self.weights[i]) + self.biases[i]
            
            if store_cache:
                self._z_values.append(z)
            
            # Apply activation
            if i == self.num_layers - 1 and self.output_activation is not None:
                # Output layer with specific activation
                current = self._activation_fn(z, self.output_activation)
            elif i == self.num_layers - 1:
                # Output layer without activation (linear)
                current = z
            else:
                # Hidden layer
                current = self._activation_fn(z, self.activation)
            
            if store_cache:
                self._activations.append(current)
        
        if single_sample:
            return current.squeeze(0)
        return current
    
    def backward(self, grad_output: np.ndarray) -> None:
        """
        Backward pass to compute gradients.
        
        Args:
            grad_output: Gradient of loss w.r.t. output
        """
        if self._input is None or not self._activations:
            raise RuntimeError("Must call forward() before backward()")
        
        # Handle single sample gradient
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(1, -1)
        
        batch_size = grad_output.shape[0]
        
        # Initialize gradient storage
        self.grad_weights = [np.zeros_like(w) for w in self.weights]
        self.grad_biases = [np.zeros_like(b) for b in self.biases]
        
        delta = grad_output
        
        # Backpropagate through layers
        for i in reversed(range(self.num_layers)):
            # Gradient w.r.t. weights and biases
            self.grad_weights[i] = np.dot(self._activations[i].T, delta) / batch_size
            self.grad_biases[i] = np.mean(delta, axis=0)
            
            if i > 0:
                # Gradient w.r.t. previous layer activation
                delta = np.dot(delta, self.weights[i].T)
                
                # Apply activation derivative
                if i == self.num_layers - 1 and self.output_activation is not None:
                    activation_grad = self._activation_derivative(self._z_values[i-1], self.output_activation)
                else:
                    activation_grad = self._activation_derivative(self._z_values[i-1], self.activation)
                
                delta = delta * activation_grad
    
    def step(self, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        """
        Update weights using Adam optimizer.
        
        Args:
            lr: Learning rate
            beta1: First moment decay rate
            beta2: Second moment decay rate  
            eps: Small constant for numerical stability
        """
        if not hasattr(self, 'grad_weights'):
            raise RuntimeError("Must call backward() before step()")
        
        self.t += 1
        
        # Bias correction terms
        bias_correction1 = 1 - beta1 ** self.t
        bias_correction2 = 1 - beta2 ** self.t
        
        for i in range(self.num_layers):
            # Update first moment estimate
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * self.grad_weights[i]
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * self.grad_biases[i]
            
            # Update second moment estimate
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (self.grad_weights[i] ** 2)
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (self.grad_biases[i] ** 2)
            
            # Bias-corrected estimates
            m_w_hat = self.m_w[i] / bias_correction1
            m_b_hat = self.m_b[i] / bias_correction1
            v_w_hat = self.v_w[i] / bias_correction2
            v_b_hat = self.v_b[i] / bias_correction2
            
            # Update weights and biases
            self.weights[i] -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
            self.biases[i] -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
    
    def zero_grad(self) -> None:
        """Clear gradients."""
        if hasattr(self, 'grad_weights'):
            del self.grad_weights
            del self.grad_biases
    
    def clip_gradients(self, max_norm: float) -> float:
        """
        Clip gradients by global norm.
        
        Args:
            max_norm: Maximum gradient norm
            
        Returns:
            Actual gradient norm before clipping
        """
        if not hasattr(self, 'grad_weights'):
            return 0.0
        
        # Compute global gradient norm
        total_norm = 0.0
        for i in range(self.num_layers):
            total_norm += np.sum(self.grad_weights[i] ** 2)
            total_norm += np.sum(self.grad_biases[i] ** 2)
        total_norm = np.sqrt(total_norm)
        
        # Clip if necessary
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-8)
            for i in range(self.num_layers):
                self.grad_weights[i] *= clip_coef
                self.grad_biases[i] *= clip_coef
        
        return total_norm
    
    def copy(self) -> "MLP":
        """Create a deep copy of the network."""
        new_mlp = MLP(self.sizes, self.activation, self.output_activation, seed=0)
        
        # Copy weights and biases
        for i in range(self.num_layers):
            new_mlp.weights[i] = self.weights[i].copy()
            new_mlp.biases[i] = self.biases[i].copy()
        
        return new_mlp
    
    def save_state(self) -> Dict[str, Any]:
        """Save network state for checkpointing."""
        return {
            'sizes': self.sizes,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'weights': [w.copy() for w in self.weights],
            'biases': [b.copy() for b in self.biases],
            'm_w': [m.copy() for m in self.m_w],
            'v_w': [v.copy() for v in self.v_w],
            'm_b': [m.copy() for m in self.m_b],
            'v_b': [v.copy() for v in self.v_b],
            't': self.t
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load network state from checkpoint."""
        self.sizes = state['sizes']
        self.activation = state['activation']
        self.output_activation = state['output_activation']
        self.num_layers = len(self.sizes) - 1
        
        self.weights = [w.copy() for w in state['weights']]
        self.biases = [b.copy() for b in state['biases']]
        self.m_w = [m.copy() for m in state['m_w']]
        self.v_w = [v.copy() for v in state['v_w']]
        self.m_b = [m.copy() for m in state['m_b']]
        self.v_b = [v.copy() for v in state['v_b']]
        self.t = state['t']