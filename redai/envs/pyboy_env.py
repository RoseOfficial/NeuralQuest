"""PyBoy Game Boy emulator environment wrapper."""

import numpy as np
from typing import Tuple, Dict, Optional, Any
import random
import time
# from ..nets.progress_detector import ProgressDetector  # Removed - not needed for speedrunning
from .pokemon_reward_system import PokemonRewardSystem
from .game_navigation_helper import NavigationHelper


class Env:
    """
    Game Boy environment using PyBoy emulator.
    
    Provides a minimal Gym-like interface with 9-action discrete space
    and RAM-based observations. Supports deterministic execution,
    frame-skip, sticky actions, and savestate management.
    """
    
    ACTIONS = ["noop", "up", "down", "left", "right", "A", "B", "start", "select"]
    
    def __init__(
        self,
        rom_path: str,
        frame_skip: int = 4,
        sticky_p: float = 0.1,
        frame_stack: int = 4,
        max_episode_steps: int = 50000,  # Increased for full game completion
        deterministic: bool = True,
        headless: bool = False,  # Enable visuals for speedrunning
        use_progress_detector: bool = False,  # Make progress detector optional for performance
        use_speedrun_rewards: bool = True,  # Enable Pokemon speedrun rewards
        seed: Optional[int] = None
    ):
        """
        Initialize PyBoy environment.
        
        Args:
            rom_path: Path to Game Boy ROM file
            frame_skip: Number of frames to repeat each action
            sticky_p: Probability of repeating previous action
            frame_stack: Number of RAM frames to stack for observation
            max_episode_steps: Maximum steps per episode
            deterministic: Enable deterministic execution
            headless: Run without display
            use_progress_detector: Enable progress detection neural network (impacts performance)
            use_speedrun_rewards: Enable Pokemon Red speedrunning reward system
            seed: Random seed for reproducibility
        """
        self.rom_path = rom_path
        self.frame_skip = frame_skip
        self.sticky_p = sticky_p
        self.frame_stack = frame_stack
        self.use_speedrun_rewards = use_speedrun_rewards
        self.max_episode_steps = max_episode_steps
        self.deterministic = deterministic
        self.headless = headless
        self.use_progress_detector = use_progress_detector
        
        self._pyboy = None
        self._last_action = 0
        self._step_count = 0
        self._ram_history = []
        self._episode_count = 0
        self._prev_obs = None
        self._current_button_state = None  # Track current pressed button for efficiency
        
        # Performance monitoring
        self._frame_count = 0
        self._start_time = time.time()
        self._last_fps_report = time.time()
        
        # Progress detector removed - using Pokemon-specific rewards instead
        self._progress_detector = None
        
        # Initialize Pokemon reward system placeholder
        self._pokemon_rewards = None
        
        if seed is not None:
            self.seed(seed)
        
        self._init_pyboy()
    
    def _init_pyboy(self) -> None:
        """Initialize PyBoy emulator."""
        try:
            from pyboy import PyBoy
        except ImportError:
            raise ImportError("PyBoy not installed. Install with: pip install pyboy")
        
        self._pyboy = PyBoy(
            self.rom_path,
            window="null" if self.headless else "SDL2",
            debug=False,
            sound=False,  # Disable sound to prevent buffer overrun crashes
            sound_emulated=False  # Disable sound emulation for performance
        )
        
        # Set unlimited emulation speed for maximum performance
        self._pyboy.set_emulation_speed(0)
        
        # Additional sound disabling for buffer overrun prevention
        try:
            if hasattr(self._pyboy, 'disable_sound'):
                self._pyboy.disable_sound()
            elif hasattr(self._pyboy, 'sound') and hasattr(self._pyboy.sound, 'disable'):
                self._pyboy.sound.disable()
        except:
            pass  # Continue if sound disabling fails
        
        # Get initial RAM state
        self._update_ram_history()
        
        # Progress detector disabled - using Pokemon-specific reward system instead
        
        # Initialize Pokemon speedrun reward system (if enabled)
        if self.use_speedrun_rewards:
            self._pokemon_rewards = PokemonRewardSystem()
            
        # Initialize navigation helper for menu sequences
        self._navigation_helper = NavigationHelper()
        self._action_history = []
    
    def _get_ram_vector(self) -> np.ndarray:
        """Get current RAM state as normalized vector (optimized for performance)."""
        if self._pyboy is None:
            raise RuntimeError("PyBoy not initialized")
        
        # Optimized: Only read the most important RAM regions for Game Boy games
        # This reduces memory access by ~80% while preserving game state information
        ram_data = []
        
        # Core game state (reduced from full 16KB+ to ~4KB)
        # Work RAM - most critical for game state (0xC000-0xDFFF)
        for addr in range(0xC000, 0xD000):  # First 4KB of work RAM (most active)
            ram_data.append(self._pyboy.memory[addr])
        
        # Sprite data (OAM) - for entity positions (0xFE00-0xFE9F) 
        for addr in range(0xFE00, 0xFEA0):  # Object Attribute Memory
            ram_data.append(self._pyboy.memory[addr])
        
        # High RAM - system state (0xFF80-0xFFFE)
        for addr in range(0xFF80, 0xFFFF):  # High RAM (127 bytes)
            ram_data.append(self._pyboy.memory[addr])
        
        # Key hardware registers for timing/state (critical for game progression)
        hardware_regs = [0xFF40, 0xFF41, 0xFF42, 0xFF43, 0xFF44, 0xFF45, 0xFF46, 0xFF47,
                        0xFF48, 0xFF49, 0xFF4A, 0xFF4B, 0xFF4C, 0xFF4D, 0xFF4E, 0xFF4F]
        for addr in hardware_regs:
            ram_data.append(self._pyboy.memory[addr])
        
        # Convert to numpy array and normalize to [0, 1]
        ram_array = np.array(ram_data, dtype=np.float32) / 255.0
        return ram_array
    
    def _update_ram_history(self) -> None:
        """Update RAM frame stack history."""
        current_ram = self._get_ram_vector()
        self._ram_history.append(current_ram)
        
        # Maintain frame stack size
        if len(self._ram_history) > self.frame_stack:
            self._ram_history.pop(0)
        
        # Pad with zeros if not enough history
        while len(self._ram_history) < self.frame_stack:
            self._ram_history.insert(0, np.zeros_like(current_ram))
    
    def _get_observation(self) -> np.ndarray:
        """Get current stacked observation."""
        return np.concatenate(self._ram_history, axis=0)
    
    def _execute_action(self, action: int) -> None:
        """Execute action on PyBoy emulator (optimized button handling)."""
        if self._pyboy is None:
            raise RuntimeError("PyBoy not initialized")
        
        # Map action index to button press
        action_name = self.ACTIONS[action]
        
        # Optimized: Only change button state if it's different from current
        if self._current_button_state != action_name:
            # Release current button if any
            if self._current_button_state is not None and self._current_button_state != "noop":
                button_map = {
                    "up": "up", "down": "down", "left": "left", "right": "right",
                    "A": "a", "B": "b", "start": "start", "select": "select"
                }
                if self._current_button_state in button_map:
                    self._pyboy.button_release(button_map[self._current_button_state])
            
            # Press the new button
            if action_name != "noop":
                button_map = {
                    "up": "up", "down": "down", "left": "left", "right": "right",
                    "A": "a", "B": "b", "start": "start", "select": "select"
                }
                if action_name in button_map:
                    self._pyboy.button_press(button_map[action_name])
            
            self._current_button_state = action_name
        
        # Advance frames with action held
        for _ in range(self.frame_skip):
            self._pyboy.tick()
            self._frame_count += 1
    
    def reset(self, *, from_state: Optional[bytes] = None) -> np.ndarray:
        """
        Reset environment to initial state or load from savestate.
        
        Args:
            from_state: Optional savestate bytes to load from
            
        Returns:
            Initial observation
        """
        if self._pyboy is None:
            self._init_pyboy()
        
        if from_state is not None:
            # Load from savestate
            self.load_state(from_state)
        else:
            # Check if we have a game start savestate to skip intro
            from pathlib import Path
            
            # Try multiple possible locations for game_start.state
            rom_path_obj = Path(self.rom_path)
            possible_paths = [
                rom_path_obj.parent / "game_start.state",  # Same directory as ROM
                rom_path_obj.parent.parent / "game_start.state",  # Parent directory
                Path.cwd() / "game_start.state",  # Current working directory
            ]
            
            game_start_path = None
            for path in possible_paths:
                if path.exists():
                    game_start_path = path
                    break
            
            if game_start_path:
                print(f"Loading game start savestate from {game_start_path}")
                with open(game_start_path, "rb") as f:
                    game_start_state = f.read()
                self.load_state(game_start_state)
            else:
                # Reset to ROM start - PyBoy doesn't have reset, so restart
                self._pyboy.stop()
                self._init_pyboy()
        
        self._step_count = 0
        self._last_action = 0
        self._ram_history.clear()
        self._episode_count += 1
        self._prev_obs = None
        self._current_button_state = None  # Reset button state
        
        # Reset Pokemon reward system if enabled
        if self.use_speedrun_rewards and hasattr(self, '_pokemon_rewards'):
            self._pokemon_rewards.reset()
            
        # Reset navigation helper
        if hasattr(self, '_navigation_helper'):
            self._navigation_helper.reset()
        self._action_history = []
        
        # Initialize observation history
        for _ in range(self.frame_stack):
            self._update_ram_history()
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Action index (0-8)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if not (0 <= action < len(self.ACTIONS)):
            raise ValueError(f"Invalid action {action}. Must be 0-{len(self.ACTIONS)-1}")
        
        # Navigation assistance for menu sequences
        original_action = action
        ram_vector = self._get_ram_vector()
        current_state = self._navigation_helper.detect_game_state(
            ram_vector, self._step_count, self._action_history[-20:] if len(self._action_history) >= 20 else self._action_history
        )
        
        # Check if we should override action for menu navigation
        if self._navigation_helper.should_override_action(current_state, action):
            suggested_action = self._navigation_helper.suggest_action(
                current_state, self._step_count, self._action_history[-20:] if len(self._action_history) >= 20 else self._action_history
            )
            if suggested_action is not None:
                action = suggested_action
                if self._step_count % 10 == 0:  # Print occasionally to avoid spam
                    print(f"NAV: {self.ACTIONS[original_action]} -> {self.ACTIONS[action]} ({current_state.value})")
        
        # Track action history
        self._action_history.append(action)
        if len(self._action_history) > 100:  # Keep last 100 actions
            self._action_history.pop(0)
        
        # Apply sticky actions
        if random.random() < self.sticky_p and self._step_count > 0:
            action = self._last_action
        
        # Execute action
        self._execute_action(action)
        self._last_action = action
        self._step_count += 1
        
        # Update observation
        self._update_ram_history()
        obs = self._get_observation()
        
        # Compute reward based on enabled systems
        reward = 0.0
        reward_breakdown = {}
        
        if self.use_speedrun_rewards and hasattr(self, '_pokemon_rewards'):
            # Use Pokemon speedrunning reward system
            ram_vector = self._get_ram_vector()
            speedrun_reward, reward_breakdown = self._pokemon_rewards.compute_reward(
                ram_vector, self._step_count
            )
            reward += speedrun_reward
            
            # Check for game completion
            if self._pokemon_rewards.is_game_completed():
                done = True
                completion_time = self._pokemon_rewards.get_completion_time()
                print(f"🎉 POKEMON RED COMPLETED! Time: {completion_time} steps")
        
        if self.use_progress_detector and self._prev_obs is not None and self._progress_detector is not None:
            # Add progress detection bonus (if both systems enabled)
            base_intrinsic = 0.1  # Small base exploration bonus
            progress_reward = self._progress_detector.compute_progress_reward(
                self._prev_obs, obs, base_intrinsic
            )
            reward += progress_reward
            reward_breakdown['progress_detector'] = progress_reward
        elif not self.use_speedrun_rewards:
            # Fallback to simple exploration reward if no reward systems enabled
            reward = 0.01  # Small constant exploration bonus
        
        # Episode termination
        done = self._step_count >= self.max_episode_steps
        
        # Performance monitoring - report FPS every 5 seconds
        current_time = time.time()
        if current_time - self._last_fps_report > 5.0:
            elapsed = current_time - self._start_time
            fps = self._frame_count / elapsed if elapsed > 0 else 0
            print(f"PyBoy Performance: {fps:.1f} FPS ({self._frame_count} frames in {elapsed:.1f}s)")
            self._last_fps_report = current_time
        
        info = {
            "episode_step": self._step_count,
            "episode_count": self._episode_count,
            "action_name": self.ACTIONS[action],
            "total_reward": reward,
            "reward_breakdown": reward_breakdown,
            "frame_count": self._frame_count,
            "fps": self._frame_count / (current_time - self._start_time) if current_time > self._start_time else 0
        }
        
        # Add Pokemon progress info if speedrun rewards enabled
        if self.use_speedrun_rewards and hasattr(self, '_pokemon_rewards'):
            info.update(self._pokemon_rewards.get_progress_summary())
        
        # Store current observation for next step
        self._prev_obs = obs.copy()
        
        return obs, reward, done, info
    
    def render_rgb(self) -> np.ndarray:
        """Get RGB screen buffer for optional credits detection."""
        if self._pyboy is None:
            raise RuntimeError("PyBoy not initialized")
        
        # Get screen buffer using PyBoy's screen.ndarray method
        try:
            screen_array = self._pyboy.screen.ndarray
            # Convert from (160, 144, 4) RGBA to (144, 160, 3) RGB
            if screen_array.shape == (160, 144, 4):
                # Transpose and remove alpha channel
                screen_rgb = screen_array.transpose(1, 0, 2)[:, :, :3]
            elif screen_array.shape == (144, 160, 4):
                # Just remove alpha channel
                screen_rgb = screen_array[:, :, :3]
            elif screen_array.shape == (160, 144, 3):
                # Transpose to correct orientation
                screen_rgb = screen_array.transpose(1, 0, 2)
            elif screen_array.shape == (144, 160, 3):
                # Already in correct format
                screen_rgb = screen_array
            else:
                print(f"Unexpected screen array shape: {screen_array.shape}")
                screen_rgb = screen_array[:, :, :3] if screen_array.shape[2] >= 3 else screen_array
            
            return screen_rgb.astype(np.uint8)
        except Exception as e:
            print(f"Warning: Could not access PyBoy screen buffer: {e}")
            return np.zeros((144, 160, 3), dtype=np.uint8)
    
    def save_state(self) -> bytes:
        """Save current emulator state."""
        if self._pyboy is None:
            raise RuntimeError("PyBoy not initialized")
        
        import io
        with io.BytesIO() as f:
            self._pyboy.save_state(f)
            f.seek(0)
            return f.read()
    
    def load_state(self, state: bytes) -> None:
        """Load emulator state from bytes."""
        if self._pyboy is None:
            raise RuntimeError("PyBoy not initialized")
        
        import io
        with io.BytesIO(state) as f:
            f.seek(0)
            self._pyboy.load_state(f)
        
        # Update RAM history after loading state
        self._ram_history.clear()
        for _ in range(self.frame_stack):
            self._update_ram_history()
    
    def seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
    
    @property
    def obs_dim(self) -> int:
        """Get observation dimension."""
        if not self._ram_history:
            # Estimate based on typical Game Boy RAM size
            # This is approximate - actual size depends on specific memory regions
            single_ram_size = (0xA000 - 0x8000) + (0xC000 - 0xA000) + (0xE000 - 0xC000) + (0xFEA0 - 0xFE00) + (0xFFFF - 0xFF80)
            return single_ram_size * self.frame_stack
        return len(self._get_observation())
    
    def get_progress_detector(self):
        """Progress detector removed - using Pokemon-specific rewards."""
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        if self._pyboy is not None:
            self._pyboy.stop()
            self._pyboy = None