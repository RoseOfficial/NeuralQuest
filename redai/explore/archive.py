"""Archive system for storing and sampling novel states."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List, Any
import heapq
import random
from .hashing import SimHasher


@dataclass
class CellRecord:
    """Record for a cell in the archive."""
    cell_id: int
    savestate: bytes
    visit_count: int
    first_seen_step: int
    last_seen_step: int
    
    def __lt__(self, other: "CellRecord") -> bool:
        """Comparison for heap operations (by visit count)."""
        return self.visit_count < other.visit_count


class Archive:
    """
    Archive system for novelty-based exploration.
    
    Maintains a collection of novel states discovered during exploration,
    with intelligent sampling for frontier-based resets and capacity management.
    """
    
    def __init__(
        self,
        input_dim: int,
        capacity: int = 20000,
        novel_lru: int = 5000,
        hamming_threshold: int = 2,
        hash_bits: int = 64,
        projection_dim: int = 64,
        seed: int = 0
    ):
        """
        Initialize archive system.
        
        Args:
            input_dim: Dimension of input observations
            capacity: Maximum number of cells to store
            novel_lru: LRU window for novelty detection
            hamming_threshold: Minimum Hamming distance for novelty
            hash_bits: Number of bits in cell hash
            projection_dim: Dimension for random projection
            seed: Random seed
        """
        self.capacity = capacity
        self.novel_lru = novel_lru
        self.hamming_threshold = hamming_threshold
        
        # Initialize hasher
        self.hasher = SimHasher(input_dim, hash_bits, seed)
        
        # Storage for cells
        self.cells: Dict[int, CellRecord] = {}
        
        # Recent cell tracking for LRU novelty detection
        self.recent_cells: List[int] = []
        
        # Statistics
        self.total_steps = 0
        self.cells_added = 0
        self.cells_evicted = 0
    
    def add_if_novel(self, obs: np.ndarray, savestate: bytes, step: int) -> bool:
        """
        Add observation to archive if novel.
        
        Args:
            obs: Observation array
            savestate: Emulator savestate bytes
            step: Current global step
            
        Returns:
            True if cell was added (novel), False otherwise
        """
        self.total_steps = step
        cell_id = self.hasher.cell_id(obs)
        
        # Check if cell already exists
        if cell_id in self.cells:
            # Update existing cell
            self.cells[cell_id].visit_count += 1
            self.cells[cell_id].last_seen_step = step
            return False
        
        # Check novelty against recent cells
        if not self._is_novel(cell_id):
            return False
        
        # Add new cell
        new_cell = CellRecord(
            cell_id=cell_id,
            savestate=savestate,
            visit_count=1,
            first_seen_step=step,
            last_seen_step=step
        )
        
        self.cells[cell_id] = new_cell
        self.cells_added += 1
        
        # Update recent cells tracking
        self.recent_cells.append(cell_id)
        if len(self.recent_cells) > self.novel_lru:
            self.recent_cells.pop(0)
        
        # Evict if over capacity
        if len(self.cells) > self.capacity:
            self._evict_cell()
        
        return True
    
    def _is_novel(self, cell_id: int) -> bool:
        """
        Check if cell ID is novel based on recent history and Hamming distance.
        
        Args:
            cell_id: Cell ID to check
            
        Returns:
            True if novel, False otherwise
        """
        # Check if seen in recent LRU window
        if cell_id in self.recent_cells[-self.novel_lru:]:
            return False
        
        # Check Hamming distance to nearby cells
        if self.hamming_threshold > 0:
            for recent_id in self.recent_cells[-min(1000, len(self.recent_cells)):]:
                if self.hasher.hamming_distance(cell_id, recent_id) < self.hamming_threshold:
                    return False
        
        return True
    
    def _evict_cell(self) -> None:
        """Evict least useful cell to maintain capacity."""
        if not self.cells:
            return
        
        # Find cell to evict (highest visit count + oldest)
        eviction_scores = []
        current_step = self.total_steps
        
        for cell_id, cell in self.cells.items():
            # Score based on visit count and age
            age_factor = (current_step - cell.last_seen_step) / max(current_step, 1)
            visit_factor = cell.visit_count
            
            # Higher score = more likely to evict
            score = visit_factor * 0.7 + age_factor * 0.3
            eviction_scores.append((score, cell_id))
        
        # Sort by score (descending) and evict highest scoring
        eviction_scores.sort(reverse=True)
        cell_to_evict = eviction_scores[0][1]
        
        del self.cells[cell_to_evict]
        self.cells_evicted += 1
        
        # Remove from recent cells if present
        if cell_to_evict in self.recent_cells:
            self.recent_cells.remove(cell_to_evict)
    
    def sample_frontier(self, top_k: int = 512, temperature: float = 1.0) -> Optional[CellRecord]:
        """
        Sample a frontier cell for resetting exploration.
        
        Args:
            top_k: Number of top frontier cells to consider
            temperature: Softmax temperature for sampling
            
        Returns:
            Selected frontier cell or None if archive is empty
        """
        if not self.cells:
            return None
        
        # Compute frontier scores
        current_step = self.total_steps
        scored_cells = []
        
        for cell in self.cells.values():
            score = self._compute_frontier_score(cell, current_step)
            scored_cells.append((score, cell))
        
        # Sort by score (descending) and take top_k
        scored_cells.sort(reverse=True, key=lambda x: x[0])
        top_cells = scored_cells[:min(top_k, len(scored_cells))]
        
        if not top_cells:
            return None
        
        # Softmax sampling over top cells
        scores = np.array([score for score, _ in top_cells])
        if temperature > 0:
            exp_scores = np.exp(scores / temperature)
            probs = exp_scores / np.sum(exp_scores)
            
            # Sample according to probabilities
            selected_idx = np.random.choice(len(top_cells), p=probs)
        else:
            # Greedy selection
            selected_idx = 0
        
        return top_cells[selected_idx][1]
    
    def _compute_frontier_score(self, cell: CellRecord, current_step: int) -> float:
        """
        Compute frontier score for a cell.
        
        Higher scores indicate better frontier cells for exploration.
        
        Args:
            cell: Cell record
            current_step: Current global step
            
        Returns:
            Frontier score
        """
        # Factors that make a good frontier cell:
        # 1. Low visit count (unexplored)
        # 2. Reasonable age (not too old, not too new)
        # 3. Depth approximation (later discoveries may be deeper)
        
        # Visit count factor (lower is better)
        visit_factor = 1.0 / (1.0 + cell.visit_count)
        
        # Age factor (prefer moderately aged cells)
        age = current_step - cell.first_seen_step
        normalized_age = age / max(current_step, 1)
        age_factor = 4 * normalized_age * (1 - normalized_age)  # Inverted U-shape
        
        # Depth approximation (later discoveries might be deeper)
        depth_factor = cell.first_seen_step / max(current_step, 1)
        
        # Combine factors
        score = (
            0.6 * visit_factor +
            0.3 * age_factor +
            0.1 * depth_factor
        )
        
        return score
    
    def size(self) -> int:
        """Get current number of cells in archive."""
        return len(self.cells)
    
    def stats(self) -> Dict[str, Any]:
        """Get archive statistics."""
        if not self.cells:
            return {
                'size': 0,
                'cells_added': self.cells_added,
                'cells_evicted': self.cells_evicted,
                'avg_visit_count': 0.0,
                'total_visits': 0,
                'oldest_cell_age': 0,
                'newest_cell_age': 0
            }
        
        visit_counts = [cell.visit_count for cell in self.cells.values()]
        ages = [self.total_steps - cell.first_seen_step for cell in self.cells.values()]
        
        return {
            'size': len(self.cells),
            'cells_added': self.cells_added,
            'cells_evicted': self.cells_evicted,
            'avg_visit_count': float(np.mean(visit_counts)),
            'total_visits': int(np.sum(visit_counts)),
            'oldest_cell_age': int(np.max(ages)) if ages else 0,
            'newest_cell_age': int(np.min(ages)) if ages else 0,
            'capacity_used': len(self.cells) / self.capacity
        }
    
    def get_frontier_cells(self, num_cells: int = 10) -> List[CellRecord]:
        """Get top frontier cells for analysis."""
        if not self.cells:
            return []
        
        scored_cells = []
        current_step = self.total_steps
        
        for cell in self.cells.values():
            score = self._compute_frontier_score(cell, current_step)
            scored_cells.append((score, cell))
        
        scored_cells.sort(reverse=True, key=lambda x: x[0])
        return [cell for _, cell in scored_cells[:num_cells]]
    
    def save_state(self) -> Dict[str, Any]:
        """Save archive state for checkpointing."""
        # Convert cells to serializable format
        cells_data = {}
        for cell_id, cell in self.cells.items():
            cells_data[str(cell_id)] = {
                'cell_id': cell.cell_id,
                'savestate': cell.savestate,
                'visit_count': cell.visit_count,
                'first_seen_step': cell.first_seen_step,
                'last_seen_step': cell.last_seen_step
            }
        
        return {
            'capacity': self.capacity,
            'novel_lru': self.novel_lru,
            'hamming_threshold': self.hamming_threshold,
            'hasher_state': self.hasher.save_state(),
            'cells': cells_data,
            'recent_cells': self.recent_cells.copy(),
            'total_steps': self.total_steps,
            'cells_added': self.cells_added,
            'cells_evicted': self.cells_evicted
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load archive state from checkpoint."""
        self.capacity = state['capacity']
        self.novel_lru = state['novel_lru']
        self.hamming_threshold = state['hamming_threshold']
        
        # Restore hasher
        self.hasher.load_state(state['hasher_state'])
        
        # Restore cells
        self.cells = {}
        for cell_id_str, cell_data in state['cells'].items():
            cell_id = int(cell_id_str)
            self.cells[cell_id] = CellRecord(
                cell_id=cell_data['cell_id'],
                savestate=cell_data['savestate'],
                visit_count=cell_data['visit_count'],
                first_seen_step=cell_data['first_seen_step'],
                last_seen_step=cell_data['last_seen_step']
            )
        
        # Restore tracking
        self.recent_cells = state['recent_cells']
        self.total_steps = state['total_steps']
        self.cells_added = state['cells_added']
        self.cells_evicted = state['cells_evicted']