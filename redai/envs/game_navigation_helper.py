"""Game navigation helper for Pokemon Red startup and menu sequences."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum


class GameState(Enum):
    """Pokemon Red game states for navigation assistance."""
    UNKNOWN = "unknown"
    TITLE_SCREEN = "title_screen"
    INTRO_SEQUENCE = "intro_sequence"
    NEW_GAME_MENU = "new_game_menu"
    NAMING_SCREEN = "naming_screen"
    INTRO_TEXT = "intro_text"
    GAMEPLAY = "gameplay"
    MENU_STUCK = "menu_stuck"


class NavigationHelper:
    """
    Helps agent navigate Pokemon Red menus and startup sequences.
    
    Provides intelligent action suggestions during menu phases to prevent
    getting stuck in startup loops.
    """
    
    def __init__(self):
        """Initialize navigation helper."""
        self.state_history = []
        self.action_history = []
        self.stuck_counter = 0
        self.last_state = GameState.UNKNOWN
        self.new_game_ensured = False  # Track if we've properly selected NEW GAME
        
        # Action mappings
        self.action_names = ["noop", "up", "down", "left", "right", "A", "B", "start", "select"]
        
        # State-specific action sequences to get through menus
        self.navigation_sequences = {
            GameState.TITLE_SCREEN: [
                {"actions": [7, 7, 7], "description": "Press START multiple times to ensure activation"},  # START
            ],
            GameState.NEW_GAME_MENU: [
                {"actions": [1, 5], "description": "Navigate to NEW GAME (UP then A)"},  # UP, A (ensure NEW GAME is selected)
                {"actions": [5, 5], "description": "Confirm NEW GAME selection"},  # A, A (double confirm)
            ],
            GameState.INTRO_SEQUENCE: [
                {"actions": [5, 5, 5, 5, 5], "description": "Skip intro sequence rapidly"},  # A multiple times
            ],
            GameState.NAMING_SCREEN: [
                {"actions": [2, 2, 2, 5], "description": "Navigate to default name and accept"},  # DOWN, DOWN, DOWN, A
                {"actions": [5, 5], "description": "Double confirm name selection"},  # A, A
            ],
            GameState.INTRO_TEXT: [
                {"actions": [5, 5, 5, 5, 5, 5], "description": "Skip all intro dialogue"},  # A multiple times
            ],
            GameState.MENU_STUCK: [
                {"actions": [6, 6, 6, 7, 1, 1, 5], "description": "B to cancel, START, UP UP, A for NEW GAME"},  # Full escape sequence
            ]
        }
        
        # Menu detection thresholds
        self.menu_stuck_threshold = 50  # Steps in menus without progress
        
    def detect_game_state(self, ram_state: np.ndarray, step_count: int, last_actions: List[int]) -> GameState:
        """
        Detect current game state from RAM and action history.
        
        Args:
            ram_state: Current RAM state from Game Boy
            step_count: Current step in episode
            last_actions: Recent actions taken
            
        Returns:
            Detected game state
        """
        # Very basic state detection based on patterns
        # In a real implementation, you'd check specific RAM addresses
        
        # If early in episode, likely in startup sequence
        if step_count < 200:  # Extended startup detection
            if step_count < 30:
                return GameState.TITLE_SCREEN
            elif step_count < 60:
                return GameState.NEW_GAME_MENU  # Critical phase - ensure NEW GAME
            elif step_count < 90:
                return GameState.INTRO_SEQUENCE
            elif step_count < 120:
                return GameState.NAMING_SCREEN
            elif step_count < 180:
                return GameState.INTRO_TEXT
            else:
                return GameState.GAMEPLAY  # Should be in game by now
        
        # Check for menu stuck patterns (be less aggressive)
        if step_count > 500 and len(last_actions) >= 20:  # Only after giving time for startup
            recent_actions = last_actions[-20:]
            # If lots of repetitive actions without progress
            menu_actions = [5, 6, 7, 8]  # A, B, START, SELECT
            menu_action_count = sum(1 for action in recent_actions if action in menu_actions)
            
            # Check for repetitive patterns (same action repeated)
            if len(set(recent_actions[-10:])) <= 2:  # Only 1-2 different actions in last 10
                return GameState.MENU_STUCK
            
            if menu_action_count > 15:  # More than 75% menu actions in larger window
                return GameState.MENU_STUCK
        
        # If step count is high, assume gameplay
        if step_count > 200:
            return GameState.GAMEPLAY
            
        return GameState.UNKNOWN
    
    def suggest_action(self, current_state: GameState, step_count: int, 
                      last_actions: List[int]) -> Optional[int]:
        """
        Suggest helpful action for current game state.
        
        Args:
            current_state: Current detected game state
            step_count: Current step in episode
            last_actions: Recent actions taken
            
        Returns:
            Suggested action index, or None if no suggestion
        """
        # Track state changes
        if current_state != self.last_state:
            self.stuck_counter = 0
            print(f"NAVIGATION: State changed to {current_state.value} at step {step_count}")
            
            # Mark NEW GAME as ensured when we move past the menu
            if current_state in [GameState.INTRO_SEQUENCE, GameState.NAMING_SCREEN]:
                self.new_game_ensured = True
                print("NAVIGATION: NEW GAME selection confirmed!")
        else:
            self.stuck_counter += 1
        
        self.last_state = current_state
        
        # Don't interfere with gameplay
        if current_state == GameState.GAMEPLAY:
            return None
            
        # Special handling for NEW GAME menu - be extra aggressive
        if current_state == GameState.NEW_GAME_MENU and not self.new_game_ensured:
            if self.stuck_counter < 5:
                print("NAVIGATION: Ensuring NEW GAME selection with UP navigation")
                return 1  # UP - move cursor to NEW GAME
            else:
                print("NAVIGATION: Confirming NEW GAME with A press")
                return 5  # A - select NEW GAME
        
        # Get navigation sequence for current state
        if current_state in self.navigation_sequences:
            sequences = self.navigation_sequences[current_state]
            
            for sequence in sequences:
                actions = sequence["actions"]
                # Use stuck counter to cycle through actions in sequence
                action_index = self.stuck_counter % len(actions)
                suggested_action = actions[action_index]
                
                if self.stuck_counter % 20 == 0:  # Print occasionally
                    print(f"NAVIGATION: {sequence['description']} (action: {self.action_names[suggested_action]})")
                
                return suggested_action
        
        # Default fallback for unknown states
        if self.stuck_counter > 30:
            # Try START button if stuck for a while
            return 7  # START
            
        return None
    
    def should_override_action(self, current_state: GameState, proposed_action: int) -> bool:
        """
        Determine if we should override the agent's proposed action.
        
        Args:
            current_state: Current game state
            proposed_action: Action the agent wants to take
            
        Returns:
            True if we should override with navigation help
        """
        # Override if we're in a menu state and agent isn't making progress
        if current_state in [GameState.TITLE_SCREEN, GameState.NEW_GAME_MENU]:
            return self.stuck_counter > 10  # Give more time before overriding
        
        if current_state in [GameState.INTRO_SEQUENCE, GameState.NAMING_SCREEN, 
                           GameState.INTRO_TEXT]:
            return self.stuck_counter > 15  # Even more time for complex sequences
        
        # Override if detected as stuck (but only occasionally to avoid spam)
        if current_state == GameState.MENU_STUCK:
            return self.stuck_counter % 10 < 3  # Override 3 out of every 10 steps
            
        return False
    
    def reset(self):
        """Reset helper state for new episode."""
        self.state_history = []
        self.action_history = []
        self.stuck_counter = 0
        self.last_state = GameState.UNKNOWN
        self.new_game_ensured = False
        print("NAVIGATION: Helper reset for new episode - will ensure NEW GAME selection")