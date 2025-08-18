"""Vectorized environment wrapper for multiple PyBoy instances."""

import numpy as np
import time
import multiprocessing as mp
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import os
from PIL import Image

from .pyboy_env import Env


@dataclass
class VectorTransition:
    """Vectorized environment transition."""
    obs: np.ndarray  # (n_envs, obs_dim)
    actions: np.ndarray  # (n_envs,)
    rewards: np.ndarray  # (n_envs,)
    dones: np.ndarray  # (n_envs,)
    infos: List[Dict[str, Any]]  # (n_envs,)


class VectorEnv:
    """
    Vectorized environment running multiple PyBoy instances in parallel.
    
    Supports running n_envs PyBoy instances where one can have visuals enabled
    while others run headless for maximum performance. All instances share
    the same neural network for training efficiency.
    """
    
    def __init__(
        self,
        rom_path: str,
        n_envs: int = 10,
        visual_env_idx: int = 0,  # Which environment should show visuals
        frame_skip: int = 4,
        sticky_p: float = 0.1,
        frame_stack: int = 4,
        max_episode_steps: int = 1500,
        deterministic: bool = True,
        use_progress_detector: bool = False,
        seed: Optional[int] = None,
        monitor_progress: bool = False,
        screenshot_dir: Optional[str] = None
    ):
        """
        Initialize vectorized PyBoy environment.
        
        Args:
            rom_path: Path to Game Boy ROM file
            n_envs: Number of parallel environments
            visual_env_idx: Index of environment to show visuals (others headless)
            frame_skip: Number of frames to repeat each action
            sticky_p: Probability of repeating previous action
            frame_stack: Number of RAM frames to stack for observation
            max_episode_steps: Maximum steps per episode
            deterministic: Enable deterministic execution
            use_progress_detector: Enable progress detection (impacts performance)
            seed: Random seed for reproducibility
            monitor_progress: Enable periodic screenshot capture for monitoring
            screenshot_dir: Directory to save screenshots (default: "progress_screenshots")
        """
        self.rom_path = rom_path
        self.n_envs = n_envs
        self.visual_env_idx = visual_env_idx
        self.frame_skip = frame_skip
        self.sticky_p = sticky_p
        self.frame_stack = frame_stack
        self.max_episode_steps = max_episode_steps
        self.deterministic = deterministic
        self.use_progress_detector = use_progress_detector
        self.seed = seed
        self.monitor_progress = monitor_progress
        self.screenshot_dir = screenshot_dir or "progress_screenshots"
        
        # Initialize environments
        self.envs: List[Env] = []
        self._setup_environments()
        
        # Get observation dimensions
        self.obs_dim = self.envs[0].obs_dim
        self.action_space_size = len(Env.ACTIONS)
        
        # Performance tracking
        self._step_count = 0
        self._start_time = time.time()
        self._last_fps_report = time.time()
        
        # Progress monitoring
        self._last_screenshot_time = time.time()
        self._screenshot_interval = 10.0  # Take screenshots every 10 seconds
        self._setup_screenshot_monitoring()
        
        # Thread pool for parallel operations
        self._executor = ThreadPoolExecutor(max_workers=n_envs)
        
        print(f"VectorEnv initialized: {n_envs} environments")
        print(f"  - Visual environment: #{visual_env_idx}")
        print(f"  - Headless environments: {n_envs - 1}")
        print(f"  - Observation dimension: {self.obs_dim}")
    
    def _setup_environments(self) -> None:
        """Setup all PyBoy environments with error handling for visual instances."""
        print("Setting up PyBoy environments...")
        
        visual_failed = False
        
        for i in range(self.n_envs):
            # Determine if this environment should show visuals
            headless = (i != self.visual_env_idx) or visual_failed or (self.visual_env_idx < 0)
            
            # Create environment with unique seed if provided
            env_seed = None if self.seed is None else self.seed + i
            
            try:
                env = Env(
                    rom_path=self.rom_path,
                    frame_skip=self.frame_skip,
                    sticky_p=self.sticky_p,
                    frame_stack=self.frame_stack,
                    max_episode_steps=self.max_episode_steps,
                    deterministic=self.deterministic,
                    headless=headless,
                    use_progress_detector=self.use_progress_detector,
                    seed=env_seed
                )
                
                self.envs.append(env)
                status = "VISUAL" if not headless else "HEADLESS"
                print(f"  Environment {i}: {status}")
                
            except Exception as e:
                if not headless and i == self.visual_env_idx:
                    # Visual environment failed, retry as headless
                    print(f"  Environment {i}: VISUAL failed ({e}), retrying as HEADLESS")
                    visual_failed = True
                    
                    try:
                        env = Env(
                            rom_path=self.rom_path,
                            frame_skip=self.frame_skip,
                            sticky_p=self.sticky_p,
                            frame_stack=self.frame_stack,
                            max_episode_steps=self.max_episode_steps,
                            deterministic=self.deterministic,
                            headless=True,  # Force headless
                            use_progress_detector=self.use_progress_detector,
                            seed=env_seed
                        )
                        
                        self.envs.append(env)
                        print(f"  Environment {i}: HEADLESS (fallback)")
                        
                    except Exception as e2:
                        print(f"  Environment {i}: FAILED completely - {e2}")
                        raise
                else:
                    print(f"  Environment {i}: FAILED - {e}")
                    raise
        
        if visual_failed:
            print("  WARNING: Visual environment failed, all environments running headless")
            self.visual_env_idx = -1  # Mark as no visual environment
    
    def _setup_screenshot_monitoring(self) -> None:
        """Setup screenshot monitoring directory and files."""
        if not self.monitor_progress:
            return
        
        # Create screenshot directory
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        # Create subdirectories for each environment
        for i in range(self.n_envs):
            env_dir = os.path.join(self.screenshot_dir, f"env_{i:02d}")
            os.makedirs(env_dir, exist_ok=True)
        
        print(f"  - Progress monitoring enabled: {self.screenshot_dir}")
        print(f"  - Screenshot interval: {self._screenshot_interval}s")
    
    def _capture_screenshots(self) -> None:
        """Capture screenshots from all environments."""
        if not self.monitor_progress:
            return
        
        current_time = time.time()
        timestamp = int(current_time)
        
        def capture_env_screenshot(env_idx):
            try:
                # Get screen buffer from environment
                screen_rgb = self.envs[env_idx].render_rgb()
                
                # Convert to PIL Image and save
                image = Image.fromarray(screen_rgb)
                env_dir = os.path.join(self.screenshot_dir, f"env_{env_idx:02d}")
                filename = f"screenshot_{timestamp}.png"
                filepath = os.path.join(env_dir, filename)
                image.save(filepath)
                
                # Also save as "latest.png" for easy viewing
                latest_path = os.path.join(env_dir, "latest.png")
                image.save(latest_path)
                
                return env_idx, True
                
            except Exception as e:
                print(f"Warning: Failed to capture screenshot from env {env_idx}: {e}")
                return env_idx, False
        
        # Capture screenshots in parallel
        futures = [self._executor.submit(capture_env_screenshot, i) for i in range(self.n_envs)]
        
        success_count = 0
        for future in futures:
            env_idx, success = future.result()
            if success:
                success_count += 1
        
        if success_count > 0:
            print(f"Captured {success_count}/{self.n_envs} screenshots at timestamp {timestamp}")
    
    def reset(self, *, env_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Reset environments and return initial observations.
        
        Args:
            env_indices: Indices of environments to reset (None = all)
            
        Returns:
            Stacked observations from all environments (n_envs, obs_dim)
        """
        if env_indices is None:
            env_indices = list(range(self.n_envs))
        
        # Reset environments in parallel
        def reset_env(idx):
            return idx, self.envs[idx].reset()
        
        futures = [self._executor.submit(reset_env, idx) for idx in env_indices]
        
        # Collect results
        observations = np.zeros((self.n_envs, self.obs_dim), dtype=np.float32)
        
        for future in futures:
            idx, obs = future.result()
            observations[idx] = obs
        
        # For environments not reset, get current observation
        for i in range(self.n_envs):
            if i not in env_indices:
                observations[i] = self.envs[i]._get_observation()
        
        return observations
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Step all environments in parallel.
        
        Args:
            actions: Actions for each environment (n_envs,)
            
        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        if len(actions) != self.n_envs:
            raise ValueError(f"Expected {self.n_envs} actions, got {len(actions)}")
        
        # Step environments in parallel with error handling
        def step_env(idx):
            try:
                return idx, self.envs[idx].step(int(actions[idx])), None
            except Exception as e:
                print(f"WARNING: Environment {idx} failed during step: {e}")
                # Return dummy values to maintain array structure
                dummy_obs = np.zeros(self.obs_dim, dtype=np.float32)
                return idx, (dummy_obs, 0.0, True, {"error": str(e), "fps": 0}), e
        
        futures = [self._executor.submit(step_env, i) for i in range(self.n_envs)]
        
        # Collect results
        observations = np.zeros((self.n_envs, self.obs_dim), dtype=np.float32)
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        infos = [{}] * self.n_envs
        
        for future in futures:
            idx, (obs, reward, done, info), error = future.result()
            observations[idx] = obs
            rewards[idx] = reward
            dones[idx] = done
            infos[idx] = info
            
            # Handle environment errors
            if error is not None:
                print(f"Environment {idx} encountered error, marking as done")
                dones[idx] = True
        
        # Update performance tracking
        self._step_count += self.n_envs
        
        # Performance reporting
        current_time = time.time()
        if current_time - self._last_fps_report > 10.0:  # Report every 10 seconds
            elapsed = current_time - self._start_time
            total_fps = sum(info.get('fps', 0) for info in infos)
            avg_fps = total_fps / self.n_envs
            print(f"VectorEnv Performance: {total_fps:.1f} total FPS, {avg_fps:.1f} avg FPS per env")
            self._last_fps_report = current_time
        
        # Screenshot capture for progress monitoring
        if (self.monitor_progress and 
            current_time - self._last_screenshot_time > self._screenshot_interval):
            self._capture_screenshots()
            self._last_screenshot_time = current_time
        
        return observations, rewards, dones, infos
    
    def get_savestates(self, env_indices: Optional[List[int]] = None) -> List[bytes]:
        """
        Get savestates from specified environments.
        
        Args:
            env_indices: Indices of environments to get savestates from
            
        Returns:
            List of savestate bytes
        """
        if env_indices is None:
            env_indices = list(range(self.n_envs))
        
        def get_savestate(idx):
            return idx, self.envs[idx].save_state()
        
        futures = [self._executor.submit(get_savestate, idx) for idx in env_indices]
        
        savestates = [None] * len(env_indices)
        for future in futures:
            idx, savestate = future.result()
            list_idx = env_indices.index(idx)
            savestates[list_idx] = savestate
        
        return savestates
    
    def load_savestates(self, savestates: List[bytes], env_indices: Optional[List[int]] = None) -> None:
        """
        Load savestates into specified environments.
        
        Args:
            savestates: List of savestate bytes
            env_indices: Indices of environments to load savestates into
        """
        if env_indices is None:
            env_indices = list(range(min(len(savestates), self.n_envs)))
        
        if len(savestates) != len(env_indices):
            raise ValueError("Number of savestates must match number of environment indices")
        
        def load_savestate(idx, savestate):
            self.envs[idx].load_state(savestate)
            return idx
        
        futures = [
            self._executor.submit(load_savestate, env_indices[i], savestates[i])
            for i in range(len(env_indices))
        ]
        
        # Wait for all loads to complete
        for future in futures:
            future.result()
    
    def seed_all(self, seed: int) -> None:
        """Seed all environments with incremental seeds."""
        for i, env in enumerate(self.envs):
            env.seed(seed + i)
    
    def close(self) -> None:
        """Clean up all environments and resources."""
        print("Closing VectorEnv...")
        
        # Close all environments
        for i, env in enumerate(self.envs):
            try:
                env.close()
                print(f"  Environment {i}: closed")
            except Exception as e:
                print(f"  Environment {i}: error closing - {e}")
        
        # Shutdown thread pool
        self._executor.shutdown(wait=True)
        print("VectorEnv closed successfully")
    
    @property
    def action_space(self):
        """Get action space size."""
        return self.action_space_size
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class AsyncVectorEnv(VectorEnv):
    """
    Asynchronous vectorized environment for maximum performance.
    
    Uses separate processes for each PyBoy instance to avoid GIL limitations
    and achieve maximum parallelism.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize async vectorized environment."""
        # For now, inherit from VectorEnv (could be extended to use multiprocessing)
        super().__init__(*args, **kwargs)
        print("AsyncVectorEnv: Using threaded implementation")
        print("  Note: For maximum performance, consider process-based parallelism")