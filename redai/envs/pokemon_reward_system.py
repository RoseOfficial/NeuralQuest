"""Pokemon Red specific reward system for speedrunning training."""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class GameProgress:
    """Track Pokemon Red game progress for speedrunning rewards."""
    badges: Set[int] = field(default_factory=set)  # Badge IDs (0-7)
    key_items: Set[str] = field(default_factory=set)  # Important items
    story_flags: Set[str] = field(default_factory=set)  # Story progression flags
    pokemon_count: int = 0  # Number of Pokemon caught
    elite_four_defeated: bool = False
    champion_defeated: bool = False
    hall_of_fame_entered: bool = False
    game_completed: bool = False
    total_steps: int = 0
    last_checkpoint_time: int = 0
    
    
class PokemonRewardSystem:
    """
    Speedrunning-focused reward system for Pokemon Red.
    
    Provides dense rewards for game progression while encouraging speed.
    """
    
    # Pokemon Red memory addresses (these need to be verified for the specific ROM)
    MEMORY_ADDRESSES = {
        # Badge flags (bit flags in memory)
        'badges': 0xD356,           # Boulder, Cascade, Thunder, Rainbow, Soul, Marsh, Volcano, Earth badges
        
        # Key story progression
        'player_id': 0xD359,        # Player trainer ID
        'money': 0xD347,           # Player money (3 bytes BCD)
        'pokedex_owned': 0xD2F7,    # Number of Pokemon owned
        'pokedex_seen': 0xD30A,     # Number of Pokemon seen
        
        # Location and map
        'current_map': 0xD35E,      # Current map ID
        'player_x': 0xD362,        # Player X position
        'player_y': 0xD361,        # Player Y position
        
        # Party Pokemon
        'party_count': 0xD163,      # Number of Pokemon in party
        'party_pokemon': 0xD164,    # Party Pokemon data starts here
        
        # Elite Four / Champion flags
        'elite_four_flags': 0xD72E, # Elite Four completion flags
        'champion_flag': 0xD72F,    # Champion defeated flag
        
        # Game state
        'game_time_hours': 0xDA40,  # Game time hours
        'game_time_minutes': 0xDA41, # Game time minutes
        'game_time_seconds': 0xDA42, # Game time seconds
        
        # Hall of Fame
        'hall_of_fame': 0xD5A0,     # Hall of Fame entry flag
    }
    
    # Reward structure for speedrunning
    REWARDS = {
        'badge': 1000.0,            # Major milestone - very high reward
        'elite_four_member': 2000.0, # Elite Four member defeated
        'champion': 5000.0,         # Champion defeated - massive reward
        'hall_of_fame': 10000.0,    # Game completion - ultimate reward
        'new_map': 10.0,            # Exploration bonus
        'pokemon_caught': 50.0,     # Team building
        'story_progress': 200.0,    # Story milestone
        'time_penalty': -0.1,       # Small penalty per step to encourage speed
        'death_penalty': -100.0,    # Penalty for Pokemon fainting/game over
        'backtrack_penalty': -5.0,  # Penalty for inefficient movement
        'menu_efficiency': 10.0,    # Bonus for quick menu navigation
    }
    
    # Map IDs for key locations (these need verification)
    KEY_LOCATIONS = {
        # Gym locations
        'pewter_gym': 0x36,
        'cerulean_gym': 0x43,
        'vermillion_gym': 0x5C,
        'celadon_gym': 0x4E,
        'saffron_gym': 0x5A,
        'fuchsia_gym': 0x4B,
        'cinnabar_gym': 0x58,
        'viridian_gym': 0x2D,
        
        # Elite Four
        'elite_four_lorelei': 0x71,
        'elite_four_bruno': 0x72,
        'elite_four_agatha': 0x73,
        'elite_four_lance': 0x74,
        'champion_blue': 0x75,
        
        # Hall of Fame
        'hall_of_fame': 0x76,
    }
    
    def __init__(self):
        """Initialize the Pokemon reward system."""
        self.progress = GameProgress()
        self.previous_progress = GameProgress()
        self.visited_maps = set()
        self.position_history = []  # For backtrack detection
        self.last_map_id = None
        self.menu_start_time = None
        
    def compute_reward(self, ram_state: np.ndarray, step_count: int) -> Tuple[float, Dict[str, float]]:
        """
        Compute speedrunning reward based on current game state.
        
        Args:
            ram_state: Current RAM state from Game Boy
            step_count: Current step count in episode
            
        Returns:
            Tuple of (total_reward, reward_breakdown)
        """
        # Update current progress
        self._update_progress(ram_state, step_count)
        
        # Calculate reward components
        reward_breakdown = {}
        total_reward = 0.0
        
        # Badge rewards - massive bonus for new badges
        new_badges = self.progress.badges - self.previous_progress.badges
        if new_badges:
            badge_reward = len(new_badges) * self.REWARDS['badge']
            reward_breakdown['badges'] = badge_reward
            total_reward += badge_reward
            print(f"🏆 NEW BADGE(S) EARNED: {new_badges} (+{badge_reward} reward)")
        
        # Elite Four and Champion
        if self.progress.elite_four_defeated and not self.previous_progress.elite_four_defeated:
            elite_reward = self.REWARDS['elite_four_member']
            reward_breakdown['elite_four'] = elite_reward
            total_reward += elite_reward
            print(f"⚡ ELITE FOUR DEFEATED! (+{elite_reward} reward)")
        
        if self.progress.champion_defeated and not self.previous_progress.champion_defeated:
            champion_reward = self.REWARDS['champion']
            reward_breakdown['champion'] = champion_reward
            total_reward += champion_reward
            print(f"👑 CHAMPION DEFEATED! (+{champion_reward} reward)")
        
        # Hall of Fame - game completion
        if self.progress.hall_of_fame_entered and not self.previous_progress.hall_of_fame_entered:
            hof_reward = self.REWARDS['hall_of_fame']
            reward_breakdown['hall_of_fame'] = hof_reward
            total_reward += hof_reward
            self.progress.game_completed = True
            print(f"🌟 HALL OF FAME! GAME COMPLETED! (+{hof_reward} reward)")
        
        # Exploration rewards
        current_map = self._read_byte(ram_state, self.MEMORY_ADDRESSES['current_map'])
        if current_map != self.last_map_id and current_map not in self.visited_maps:
            map_reward = self.REWARDS['new_map']
            reward_breakdown['exploration'] = map_reward
            total_reward += map_reward
            self.visited_maps.add(current_map)
        self.last_map_id = current_map
        
        # Pokemon collection bonus
        new_pokemon = self.progress.pokemon_count - self.previous_progress.pokemon_count
        if new_pokemon > 0:
            pokemon_reward = new_pokemon * self.REWARDS['pokemon_caught']
            reward_breakdown['pokemon'] = pokemon_reward
            total_reward += pokemon_reward
        
        # Time penalty to encourage speed
        time_penalty = self.REWARDS['time_penalty']
        reward_breakdown['time_penalty'] = time_penalty
        total_reward += time_penalty
        
        # Backtracking penalty (if returning to recent positions)
        player_x = self._read_byte(ram_state, self.MEMORY_ADDRESSES['player_x'])
        player_y = self._read_byte(ram_state, self.MEMORY_ADDRESSES['player_y'])
        current_pos = (current_map, player_x, player_y)
        
        # If we have no valid game state, provide small exploration bonus instead of harsh penalties
        if current_pos == (0, 0, 0) and len(reward_breakdown) <= 1:  # Only time penalty
            exploration_bonus = 0.1  # Small positive reward for exploration when no game state
            reward_breakdown['exploration'] = exploration_bonus
            total_reward += exploration_bonus
        
        # Keep history of last 20 positions
        self.position_history.append(current_pos)
        if len(self.position_history) > 20:
            self.position_history.pop(0)
        
        # Check if current position was visited recently (backtracking)
        # Only apply backtrack penalty if we have valid position data (not all zeros)
        if (len(self.position_history) > 5 and 
            current_pos in self.position_history[:-5] and 
            current_pos != (0, 0, 0)):  # Don't penalize if position data is invalid
            backtrack_penalty = self.REWARDS['backtrack_penalty']
            reward_breakdown['backtrack'] = backtrack_penalty
            total_reward += backtrack_penalty
        
        # Store previous progress for next comparison
        self.previous_progress = GameProgress(
            badges=self.progress.badges.copy(),
            key_items=self.progress.key_items.copy(),
            story_flags=self.progress.story_flags.copy(),
            pokemon_count=self.progress.pokemon_count,
            elite_four_defeated=self.progress.elite_four_defeated,
            champion_defeated=self.progress.champion_defeated,
            hall_of_fame_entered=self.progress.hall_of_fame_entered,
            game_completed=self.progress.game_completed,
            total_steps=self.progress.total_steps
        )
        
        return total_reward, reward_breakdown
    
    def _update_progress(self, ram_state: np.ndarray, step_count: int):
        """Update current game progress from RAM state."""
        self.progress.total_steps = step_count
        
        # Read badge flags (8 bits for 8 badges)
        badge_byte = self._read_byte(ram_state, self.MEMORY_ADDRESSES['badges'])
        self.progress.badges = set()
        for i in range(8):
            if badge_byte & (1 << i):
                self.progress.badges.add(i)
        
        # Read Pokemon count
        self.progress.pokemon_count = self._read_byte(ram_state, self.MEMORY_ADDRESSES['pokedex_owned'])
        
        # Check Elite Four and Champion flags
        elite_flags = self._read_byte(ram_state, self.MEMORY_ADDRESSES['elite_four_flags'])
        self.progress.elite_four_defeated = elite_flags == 0xFF  # All Elite Four defeated
        
        champion_flag = self._read_byte(ram_state, self.MEMORY_ADDRESSES['champion_flag'])
        self.progress.champion_defeated = champion_flag > 0
        
        # Hall of Fame check
        hof_flag = self._read_byte(ram_state, self.MEMORY_ADDRESSES['hall_of_fame'])
        self.progress.hall_of_fame_entered = hof_flag > 0
        
        # Game completion check
        self.progress.game_completed = (
            self.progress.champion_defeated and 
            self.progress.hall_of_fame_entered
        )
    
    def _read_byte(self, ram_state: np.ndarray, address: int) -> int:
        """Read a single byte from RAM state."""
        if address < len(ram_state):
            return int(ram_state[address])
        return 0
    
    def _read_word(self, ram_state: np.ndarray, address: int) -> int:
        """Read a 16-bit word from RAM state (little endian)."""
        if address + 1 < len(ram_state):
            return int(ram_state[address]) + (int(ram_state[address + 1]) << 8)
        return 0
    
    def get_progress_summary(self) -> Dict[str, any]:
        """Get current progress summary for logging."""
        return {
            'badges_count': len(self.progress.badges),
            'badges_earned': list(self.progress.badges),
            'pokemon_count': self.progress.pokemon_count,
            'elite_four_defeated': self.progress.elite_four_defeated,
            'champion_defeated': self.progress.champion_defeated,
            'hall_of_fame': self.progress.hall_of_fame_entered,
            'game_completed': self.progress.game_completed,
            'maps_visited': len(self.visited_maps),
            'total_steps': self.progress.total_steps
        }
    
    def is_game_completed(self) -> bool:
        """Check if the game has been completed."""
        return self.progress.game_completed
    
    def get_completion_time(self) -> int:
        """Get total steps to completion (speedrun metric)."""
        return self.progress.total_steps if self.progress.game_completed else -1
    
    def reset(self):
        """Reset the reward system for a new episode."""
        self.progress = GameProgress()
        self.previous_progress = GameProgress()
        self.visited_maps = set()
        self.position_history = []
        self.last_map_id = None
        self.menu_start_time = None