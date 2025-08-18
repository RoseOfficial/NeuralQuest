"""PyBoy Game Boy emulator environment wrapper."""

import numpy as np
from typing import Tuple, Dict, Optional, Any
import random
from ..nets.progress_detector import ProgressDetector


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
        max_episode_steps: int = 10000,
        deterministic: bool = True,
        headless: bool = True,
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
            seed: Random seed for reproducibility
        """
        self.rom_path = rom_path
        self.frame_skip = frame_skip
        self.sticky_p = sticky_p
        self.frame_stack = frame_stack
        self.max_episode_steps = max_episode_steps
        self.deterministic = deterministic
        self.headless = headless
        
        self._pyboy = None
        self._last_action = 0
        self._step_count = 0
        self._ram_history = []
        self._episode_count = 0
        self._prev_obs = None
        
        # Initialize progress detector for learned progress
        self._progress_detector = None
        
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
            debug=False
        )
        
        if self.deterministic:
            # Enable deterministic mode
            self._pyboy.set_emulation_speed(0)  # Unlimited speed
        
        # Get initial RAM state
        self._update_ram_history()
        
        # Initialize progress detector after we know observation dimension
        if self._progress_detector is None:
            obs_dim = len(self._get_observation())
            self._progress_detector = ProgressDetector(
                input_dim=obs_dim,
                hidden_dim=256,
                learning_rate=3e-4,
                seed=42
            )
    
    def _get_ram_vector(self) -> np.ndarray:
        """Get current RAM state as normalized vector."""
        if self._pyboy is None:
            raise RuntimeError("PyBoy not initialized")
        
        # Get full RAM dump
        ram_data = []
        
        # Work RAM (0x8000-0x9FFF, 0xA000-0xBFFF, 0xC000-0xDFFF, 0xFE00-0xFE9F, 0xFF80-0xFFFE)
        for addr in range(0x8000, 0xA000):  # Video RAM
            ram_data.append(self._pyboy.memory[addr])
        for addr in range(0xA000, 0xC000):  # External RAM
            ram_data.append(self._pyboy.memory[addr])
        for addr in range(0xC000, 0xE000):  # Work RAM
            ram_data.append(self._pyboy.memory[addr])
        for addr in range(0xFE00, 0xFEA0):  # OAM
            ram_data.append(self._pyboy.memory[addr])
        for addr in range(0xFF80, 0xFFFF):  # High RAM
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
        """Execute action on PyBoy emulator."""
        if self._pyboy is None:
            raise RuntimeError("PyBoy not initialized")
        
        # Map action index to button press
        action_name = self.ACTIONS[action]
        
        # Release all buttons first
        self._pyboy.button_release("a")
        self._pyboy.button_release("b")
        self._pyboy.button_release("start")
        self._pyboy.button_release("select")
        self._pyboy.button_release("up")
        self._pyboy.button_release("down")
        self._pyboy.button_release("left")
        self._pyboy.button_release("right")
        
        # Press the selected button
        if action_name == "up":
            self._pyboy.button_press("up")
        elif action_name == "down":
            self._pyboy.button_press("down")
        elif action_name == "left":
            self._pyboy.button_press("left")
        elif action_name == "right":
            self._pyboy.button_press("right")
        elif action_name == "A":
            self._pyboy.button_press("a")
        elif action_name == "B":
            self._pyboy.button_press("b")
        elif action_name == "start":
            self._pyboy.button_press("start")
        elif action_name == "select":
            self._pyboy.button_press("select")
        # "noop" requires no button press
        
        # Advance frames with action held
        for _ in range(self.frame_skip):
            self._pyboy.tick()
    
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
        
        # Compute progress-aware reward
        reward = 0.0
        if self._prev_obs is not None and self._progress_detector is not None:
            # Base intrinsic reward (will be enhanced by progress detector)
            base_intrinsic = 0.1  # Small base exploration bonus
            
            # Get progress-enhanced reward
            reward = self._progress_detector.compute_progress_reward(
                self._prev_obs, obs, base_intrinsic
            )
        
        # Episode termination
        done = self._step_count >= self.max_episode_steps
        
        info = {
            "episode_step": self._step_count,
            "episode_count": self._episode_count,
            "action_name": self.ACTIONS[action],
            "progress_reward": reward
        }
        
        # Store current observation for next step
        self._prev_obs = obs.copy()
        
        return obs, reward, done, info
    
    def render_rgb(self) -> np.ndarray:
        """Get RGB screen buffer for optional credits detection."""
        if self._pyboy is None:
            raise RuntimeError("PyBoy not initialized")
        
        screen_buffer = self._pyboy.screen_buffer()
        return np.array(screen_buffer)
    
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
    
    def get_progress_detector(self) -> Optional[ProgressDetector]:
        """Get progress detector for training updates."""
        return self._progress_detector
    
    def close(self) -> None:
        """Clean up resources."""
        if self._pyboy is not None:
            self._pyboy.stop()
            self._pyboy = None