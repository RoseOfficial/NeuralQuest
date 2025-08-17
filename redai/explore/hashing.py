"""SimHash implementation for state discretization."""

import numpy as np
from typing import Optional


class SimHasher:
    """
    SimHash implementation for mapping high-dimensional observations to discrete cells.
    
    Uses random Gaussian projection followed by sign-based hashing to create
    64-bit binary codes that preserve similarity in the original space.
    """
    
    def __init__(self, input_dim: int, hash_bits: int = 64, seed: int = 0):
        """
        Initialize SimHasher with random projection matrix.
        
        Args:
            input_dim: Dimension of input observations
            hash_bits: Number of bits in output hash (typically 64)
            seed: Random seed for reproducible projection matrix
        """
        self.input_dim = input_dim
        self.hash_bits = hash_bits
        self.seed = seed
        
        # Generate random projection matrix
        np.random.seed(seed)
        self.projection_matrix = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=(input_dim, hash_bits)
        )
        
        # Optional bias term for projection
        self.bias = np.random.normal(0.0, 0.1, hash_bits)
    
    def cell_id(self, obs: np.ndarray) -> int:
        """
        Compute 64-bit cell ID for observation.
        
        Args:
            obs: Observation array of shape (input_dim,)
            
        Returns:
            64-bit integer cell ID
        """
        if obs.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {obs.shape[-1]}")
        
        # Handle batch input
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        
        # Project to hash_bits dimensions
        projected = np.dot(obs, self.projection_matrix) + self.bias
        
        # Apply sign function to get binary codes
        binary_codes = (projected >= 0).astype(np.int32)
        
        # Convert binary array to integer
        if binary_codes.shape[0] == 1:
            return self._binary_to_int(binary_codes[0])
        else:
            return [self._binary_to_int(code) for code in binary_codes]
    
    def _binary_to_int(self, binary_array: np.ndarray) -> int:
        """Convert binary array to integer."""
        # Ensure we don't exceed 64 bits for standard int
        if len(binary_array) > 64:
            binary_array = binary_array[:64]
        
        result = 0
        for i, bit in enumerate(binary_array):
            if bit:
                result |= (1 << i)
        
        return result
    
    def hamming_distance(self, id1: int, id2: int) -> int:
        """
        Compute Hamming distance between two cell IDs.
        
        Args:
            id1: First cell ID
            id2: Second cell ID
            
        Returns:
            Number of differing bits
        """
        xor_result = id1 ^ id2
        return bin(xor_result).count('1')
    
    def similarity(self, obs1: np.ndarray, obs2: np.ndarray) -> float:
        """
        Compute similarity between two observations based on Hamming distance.
        
        Args:
            obs1: First observation
            obs2: Second observation
            
        Returns:
            Similarity score in [0, 1] where 1 is identical
        """
        id1 = self.cell_id(obs1)
        id2 = self.cell_id(obs2)
        hamming_dist = self.hamming_distance(id1, id2)
        
        # Convert to similarity (1 - normalized_distance)
        return 1.0 - (hamming_dist / self.hash_bits)
    
    def get_projection_stats(self) -> dict:
        """Get statistics about the projection matrix."""
        return {
            'input_dim': self.input_dim,
            'hash_bits': self.hash_bits,
            'projection_mean': float(np.mean(self.projection_matrix)),
            'projection_std': float(np.std(self.projection_matrix)),
            'bias_mean': float(np.mean(self.bias)),
            'bias_std': float(np.std(self.bias))
        }
    
    def save_state(self) -> dict:
        """Save hasher state for checkpointing."""
        return {
            'input_dim': self.input_dim,
            'hash_bits': self.hash_bits,
            'seed': self.seed,
            'projection_matrix': self.projection_matrix.copy(),
            'bias': self.bias.copy()
        }
    
    def load_state(self, state: dict) -> None:
        """Load hasher state from checkpoint."""
        self.input_dim = state['input_dim']
        self.hash_bits = state['hash_bits']
        self.seed = state['seed']
        self.projection_matrix = state['projection_matrix'].copy()
        self.bias = state['bias'].copy()