"""Checkpoint system for Pokemon Red speedrunning training."""

import os
import pickle
import gzip
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class Checkpoint:
    """Represents a game state checkpoint."""
    name: str
    description: str
    savestate: bytes
    timestamp: float
    step_count: int
    badges_earned: List[int] = field(default_factory=list)
    pokemon_count: int = 0
    story_progress: str = ""
    difficulty: str = "normal"  # easy, normal, hard
    priority: float = 1.0  # Higher = more likely to be selected


class CheckpointManager:
    """
    Manages game state checkpoints for curriculum learning and efficient training.
    
    Allows the agent to practice specific segments of the game by loading
    from strategic savepoints rather than always starting from the beginning.
    """
    
    # Predefined checkpoint locations for Pokemon Red speedrunning
    CHECKPOINT_DEFINITIONS = {
        # Early game checkpoints
        "game_start": {
            "description": "Game beginning - choose starter Pokemon",
            "badges": [],
            "difficulty": "easy",
            "priority": 0.5
        },
        "route_1": {
            "description": "After getting starter, heading to Viridian City",
            "badges": [],
            "difficulty": "easy", 
            "priority": 0.3
        },
        "viridian_city": {
            "description": "First city visit, get Pokedex",
            "badges": [],
            "difficulty": "easy",
            "priority": 0.4
        },
        
        # Gym progression checkpoints
        "pre_brock": {
            "description": "Before Brock fight - first gym challenge",
            "badges": [],
            "difficulty": "normal",
            "priority": 1.0
        },
        "post_brock": {
            "description": "After defeating Brock - Boulder Badge earned",
            "badges": [0],
            "difficulty": "normal",
            "priority": 1.2
        },
        "pre_misty": {
            "description": "Before Misty fight - second gym",
            "badges": [0],
            "difficulty": "normal", 
            "priority": 1.0
        },
        "post_misty": {
            "description": "After defeating Misty - Cascade Badge",
            "badges": [0, 1],
            "difficulty": "normal",
            "priority": 1.2
        },
        
        # Mid-game checkpoints
        "ss_anne": {
            "description": "S.S. Anne section - important story segment",
            "badges": [0, 1],
            "difficulty": "normal",
            "priority": 0.8
        },
        "pre_lt_surge": {
            "description": "Before Lt. Surge - third gym",
            "badges": [0, 1],
            "difficulty": "normal",
            "priority": 1.0
        },
        "post_lt_surge": {
            "description": "After Lt. Surge - Thunder Badge",
            "badges": [0, 1, 2],
            "difficulty": "normal",
            "priority": 1.2
        },
        "rocket_hideout": {
            "description": "Team Rocket hideout section",
            "badges": [0, 1, 2],
            "difficulty": "hard",
            "priority": 0.9
        },
        "pokemon_tower": {
            "description": "Pokemon Tower - Lavender Town",
            "badges": [0, 1, 2],
            "difficulty": "hard",
            "priority": 0.8
        },
        
        # Late game checkpoints
        "pre_elite_four": {
            "description": "Before Elite Four - all badges earned",
            "badges": [0, 1, 2, 3, 4, 5, 6, 7],
            "difficulty": "hard",
            "priority": 2.0  # Very important for endgame practice
        },
        "elite_four_start": {
            "description": "At Elite Four entrance",
            "badges": [0, 1, 2, 3, 4, 5, 6, 7],
            "difficulty": "hard",
            "priority": 2.5
        },
        "champion_battle": {
            "description": "About to face Champion",
            "badges": [0, 1, 2, 3, 4, 5, 6, 7],
            "difficulty": "hard",
            "priority": 3.0  # Highest priority - final boss practice
        }
    }
    
    def __init__(self, checkpoint_dir: str = "pokemon_checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.checkpoints: Dict[str, Checkpoint] = {}
        self.load_all_checkpoints()
        
        # Curriculum learning parameters
        self.difficulty_weights = {"easy": 0.2, "normal": 1.0, "hard": 0.8}
        self.current_success_rates = {}  # Track success from each checkpoint
        self.checkpoint_usage_counts = {}  # Track how often each is used
        
    def save_checkpoint(
        self, 
        name: str, 
        savestate: bytes, 
        step_count: int,
        badges_earned: List[int],
        pokemon_count: int = 0,
        story_progress: str = "",
        custom_description: Optional[str] = None
    ) -> None:
        """
        Save a new checkpoint.
        
        Args:
            name: Checkpoint name
            savestate: Game savestate bytes
            step_count: Current step count
            badges_earned: List of badge IDs earned
            pokemon_count: Number of Pokemon caught
            story_progress: Story progress description
            custom_description: Custom description (overrides default)
        """
        import time
        
        # Get checkpoint definition or create default
        checkpoint_def = self.CHECKPOINT_DEFINITIONS.get(name, {
            "description": custom_description or f"Custom checkpoint: {name}",
            "badges": badges_earned,
            "difficulty": "normal",
            "priority": 1.0
        })
        
        checkpoint = Checkpoint(
            name=name,
            description=custom_description or checkpoint_def["description"],
            savestate=savestate,
            timestamp=time.time(),
            step_count=step_count,
            badges_earned=badges_earned,
            pokemon_count=pokemon_count,
            story_progress=story_progress,
            difficulty=checkpoint_def["difficulty"],
            priority=checkpoint_def["priority"]
        )
        
        self.checkpoints[name] = checkpoint
        
        # Save to disk (compressed)
        checkpoint_file = self.checkpoint_dir / f"{name}.checkpoint"
        with gzip.open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Saved checkpoint '{name}': {checkpoint.description}")
        
        # Initialize tracking
        if name not in self.current_success_rates:
            self.current_success_rates[name] = 0.5  # Neutral starting point
        if name not in self.checkpoint_usage_counts:
            self.checkpoint_usage_counts[name] = 0
    
    def load_checkpoint(self, name: str) -> Optional[Checkpoint]:
        """
        Load a specific checkpoint.
        
        Args:
            name: Checkpoint name
            
        Returns:
            Checkpoint object or None if not found
        """
        if name in self.checkpoints:
            return self.checkpoints[name]
        
        # Try to load from disk
        checkpoint_file = self.checkpoint_dir / f"{name}.checkpoint"
        if checkpoint_file.exists():
            try:
                with gzip.open(checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                    self.checkpoints[name] = checkpoint
                    return checkpoint
            except Exception as e:
                print(f"WARNING: Failed to load checkpoint '{name}': {e}")
        
        return None
    
    def load_all_checkpoints(self) -> None:
        """Load all available checkpoints from disk."""
        for checkpoint_file in self.checkpoint_dir.glob("*.checkpoint"):
            name = checkpoint_file.stem
            self.load_checkpoint(name)
    
    def select_checkpoint(
        self, 
        curriculum_stage: str = "mixed",
        exclude_used_recently: bool = True
    ) -> Optional[Checkpoint]:
        """
        Select a checkpoint for training based on curriculum learning strategy.
        
        Args:
            curriculum_stage: "easy", "normal", "hard", "mixed", "adaptive"
            exclude_used_recently: Avoid checkpoints used very recently
            
        Returns:
            Selected checkpoint or None
        """
        if not self.checkpoints:
            print("WARNING: No checkpoints available")
            return None
        
        available_checkpoints = list(self.checkpoints.values())
        
        # Filter by curriculum stage
        if curriculum_stage != "mixed":
            available_checkpoints = [
                cp for cp in available_checkpoints 
                if cp.difficulty == curriculum_stage
            ]
        
        # Exclude recently used checkpoints (simple round-robin)
        if exclude_used_recently and len(available_checkpoints) > 1:
            # Find least recently used checkpoints
            usage_counts = {cp.name: self.checkpoint_usage_counts.get(cp.name, 0) 
                          for cp in available_checkpoints}
            min_usage = min(usage_counts.values())
            available_checkpoints = [
                cp for cp in available_checkpoints
                if usage_counts[cp.name] <= min_usage + 1
            ]
        
        if not available_checkpoints:
            # Fallback to any available checkpoint
            available_checkpoints = list(self.checkpoints.values())
        
        # Adaptive selection based on success rates and priority
        if curriculum_stage == "adaptive":
            weights = []
            for cp in available_checkpoints:
                success_rate = self.current_success_rates.get(cp.name, 0.5)
                difficulty_weight = self.difficulty_weights.get(cp.difficulty, 1.0)
                
                # Higher weight for lower success rates (need more practice)
                adaptive_weight = (1.0 - success_rate) * cp.priority * difficulty_weight
                weights.append(adaptive_weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                selected_idx = np.random.choice(len(available_checkpoints), p=weights)
                selected = available_checkpoints[selected_idx]
            else:
                selected = np.random.choice(available_checkpoints)
        else:
            # Weighted random selection based on priority
            priorities = [cp.priority for cp in available_checkpoints]
            total_priority = sum(priorities)
            if total_priority > 0:
                weights = [p / total_priority for p in priorities]
                selected_idx = np.random.choice(len(available_checkpoints), p=weights)
                selected = available_checkpoints[selected_idx]
            else:
                selected = np.random.choice(available_checkpoints)
        
        # Update usage tracking
        self.checkpoint_usage_counts[selected.name] = \
            self.checkpoint_usage_counts.get(selected.name, 0) + 1
        
        print(f"Selected checkpoint: {selected.name} - {selected.description}")
        return selected
    
    def update_success_rate(self, checkpoint_name: str, success: bool) -> None:
        """
        Update success rate for a checkpoint (for adaptive curriculum).
        
        Args:
            checkpoint_name: Name of the checkpoint
            success: Whether training from this checkpoint was successful
        """
        if checkpoint_name not in self.current_success_rates:
            self.current_success_rates[checkpoint_name] = 0.5
        
        # Exponential moving average for success rate
        alpha = 0.1  # Learning rate for success tracking
        current_rate = self.current_success_rates[checkpoint_name]
        new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate
        self.current_success_rates[checkpoint_name] = new_rate
        
        print(f"Updated {checkpoint_name} success rate: {new_rate:.3f}")
    
    def get_checkpoint_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all checkpoints."""
        stats = {}
        for name, checkpoint in self.checkpoints.items():
            stats[name] = {
                "description": checkpoint.description,
                "difficulty": checkpoint.difficulty,
                "priority": checkpoint.priority,
                "badges": len(checkpoint.badges_earned),
                "success_rate": self.current_success_rates.get(name, 0.5),
                "usage_count": self.checkpoint_usage_counts.get(name, 0),
                "timestamp": checkpoint.timestamp
            }
        return stats
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoint names."""
        return list(self.checkpoints.keys())
    
    def delete_checkpoint(self, name: str) -> bool:
        """
        Delete a checkpoint.
        
        Args:
            name: Checkpoint name
            
        Returns:
            True if deleted successfully
        """
        # Remove from memory
        if name in self.checkpoints:
            del self.checkpoints[name]
        
        # Remove file
        checkpoint_file = self.checkpoint_dir / f"{name}.checkpoint"
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                print(f"Deleted checkpoint '{name}'")
                return True
            except Exception as e:
                print(f"WARNING: Failed to delete checkpoint '{name}': {e}")
                return False
        
        return True