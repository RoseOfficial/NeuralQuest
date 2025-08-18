"""Vectorized trainer for multiple PyBoy instances with shared neural network."""

import numpy as np
import os
import csv
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..envs.vector_env import VectorEnv, VectorTransition
from ..envs.pyboy_env import Env  # For ACTIONS
from ..algo.a2c_numpy import ActorCritic, compute_gae
from ..nets.rnd_numpy import RND
from ..explore.archive import Archive
from .config import Config


@dataclass
class VectorTransitionBatch:
    """Batch of vectorized environment transitions."""
    obs: np.ndarray  # (batch_size, obs_dim)
    actions: np.ndarray  # (batch_size,)
    rewards: np.ndarray  # (batch_size,)
    values: np.ndarray  # (batch_size,)
    log_probs: np.ndarray  # (batch_size,)
    dones: np.ndarray  # (batch_size,)
    infos: List[Dict[str, Any]]  # (batch_size,)


class VectorTrainer:
    """
    Vectorized trainer for NeuralQuest agent with multiple PyBoy instances.
    
    Runs multiple PyBoy environments in parallel, all sharing the same neural
    network for efficient training. Supports one visual environment and multiple
    headless environments for optimal performance.
    """
    
    def __init__(self, config: Config, n_envs: int = 10, visual_env_idx: int = 0, monitor_progress: bool = False, track_events: bool = False, event_log_dir: str = "pokemon_events"):
        """
        Initialize vectorized trainer.
        
        Args:
            config: Training configuration
            n_envs: Number of parallel environments
            visual_env_idx: Index of environment to show visuals
            monitor_progress: Enable progress monitoring with screenshots
            track_events: Enable Pokemon event tracking
            event_log_dir: Directory to save event logs
        """
        self.config = config
        self.n_envs = n_envs
        self.visual_env_idx = visual_env_idx
        self.monitor_progress = monitor_progress
        self.track_events = track_events
        self.event_log_dir = event_log_dir
        config.ensure_dirs()
        
        # Setup logger
        self.logger = logging.getLogger(f"{__name__}.VectorTrainer")
        self.logger.setLevel(logging.INFO)  # Set to INFO to reduce verbosity
        
        # Initialize vectorized environment
        self.env: Optional[VectorEnv] = None
        
        # Initialize networks (shared across all environments)
        self.actor_critic: Optional[ActorCritic] = None
        self.rnd: Optional[RND] = None
        self.archive: Optional[Archive] = None
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.total_reward = 0.0
        
        # Per-environment episode tracking
        self.env_episode_rewards = [[] for _ in range(n_envs)]
        self.env_episode_lengths = [[] for _ in range(n_envs)]
        self.env_step_counts = np.zeros(n_envs, dtype=int)
        
        # Metrics tracking
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Setup logging
        self.setup_logging()
    
    def setup_environment(self, rom_path: str) -> None:
        """
        Setup vectorized environment and initialize networks.
        
        Args:
            rom_path: Path to Game Boy ROM file
        """
        print(f"Setting up vectorized environment with ROM: {rom_path}")
        
        # Initialize vectorized environment
        self.env = VectorEnv(
            rom_path=rom_path,
            n_envs=self.n_envs,
            visual_env_idx=self.visual_env_idx,
            frame_skip=self.config.env.frame_skip,
            sticky_p=self.config.env.sticky_p,
            max_episode_steps=self.config.env.max_episode_steps,
            deterministic=self.config.env.deterministic,
            use_progress_detector=getattr(self.config.env, 'use_progress_detector', False),
            seed=self.config.env.seed,
            monitor_progress=self.monitor_progress,
            track_events=self.track_events,
            event_log_dir=self.event_log_dir
        )
        
        # Get observation dimension
        obs_dim = self.env.obs_dim
        print(f"Observation dimension: {obs_dim}")
        
        # Initialize shared networks
        self.actor_critic = ActorCritic(
            input_dim=obs_dim,
            n_actions=len(Env.ACTIONS),
            hidden_dim=self.config.algo.hidden_dim,
            seed=self.config.env.seed
        )
        
        self.rnd = RND(
            input_dim=obs_dim,
            hidden_dim=self.config.rnd.hidden_dim,
            reward_clip=self.config.rnd.reward_clip,
            norm_ema=self.config.rnd.norm_ema,
            seed=self.config.env.seed
        )
        
        # Archive is shared across all environments
        self.archive = Archive(
            input_dim=obs_dim,
            capacity=self.config.archive.capacity * self.n_envs,  # Scale capacity
            novel_lru=self.config.archive.novel_lru,
            hamming_threshold=self.config.archive.hamming_threshold,
            hash_bits=self.config.archive.hash_bits,
            projection_dim=self.config.archive.projection_dim,
            seed=self.config.env.seed
        )
        
        print("Networks initialized successfully")
        print(f"Archive capacity scaled to: {self.archive.capacity}")
    
    def setup_logging(self) -> None:
        """Setup CSV logging for metrics."""
        self.metrics_file = os.path.join(self.config.train.log_dir, "vector_metrics.csv")
        
        # Create CSV header
        headers = [
            "epoch", "global_step", "episode_count", "total_fps", "avg_fps_per_env",
            "episode_reward_mean", "episode_length_mean",
            "policy_loss", "value_loss", "entropy", "total_loss",
            "rnd_loss", "intrinsic_reward_mean",
            "archive_size", "archive_novel_rate", "cells_per_hour",
            "policy_grad_norm", "value_grad_norm",
            "n_envs", "visual_env_idx"
        ]
        
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def rollout(self) -> VectorTransitionBatch:
        """
        Collect vectorized rollout from all environments.
        
        Returns:
            Batch of transitions from all environments
        """
        if self.env is None or self.actor_critic is None:
            raise RuntimeError("Environment and networks must be initialized")
        
        # Calculate steps per environment for balanced collection
        steps_per_env = self.config.algo.batch_horizon // self.n_envs
        if steps_per_env < 1:
            steps_per_env = 1
        
        total_steps = steps_per_env * self.n_envs
        
        # Initialize storage
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_values = []
        batch_log_probs = []
        batch_dones = []
        batch_infos = []
        
        # Reset environments if this is the first rollout
        if self.global_step == 0:
            observations = self.env.reset()
            self.logger.debug("First rollout - reset all environments")
        else:
            # Get current observations from VectorEnv's stored state
            if self.env.last_observations is not None:
                observations = self.env.last_observations.copy()
                self.logger.debug(f"Continuing rollout from stored observations at global step {self.global_step}")
            else:
                # Fallback: reset if we don't have stored observations
                observations = self.env.reset()
                self.logger.debug("No stored observations - reset all environments")
        
        self.logger.debug(f"Starting vectorized rollout: steps_per_env={steps_per_env}, total_steps={total_steps}")
        
        for step_idx in range(steps_per_env):
            # Get actions from policy for all environments
            actions = np.zeros(self.n_envs, dtype=int)
            log_probs = np.zeros(self.n_envs, dtype=np.float32)
            values = np.zeros(self.n_envs, dtype=np.float32)
            
            for env_idx in range(self.n_envs):
                action, log_prob, value, aux_info = self.actor_critic.act(observations[env_idx])
                actions[env_idx] = action
                log_probs[env_idx] = log_prob
                values[env_idx] = value
            
            # Step all environments
            next_observations, env_rewards, dones, infos = self.env.step(actions)
            
            # Compute intrinsic rewards for all environments
            intrinsic_rewards = np.zeros(self.n_envs, dtype=np.float32)
            for env_idx in range(self.n_envs):
                intrinsic_rewards[env_idx] = self.rnd.intrinsic_reward(next_observations[env_idx])
            
            total_rewards = self.config.rnd.beta * intrinsic_rewards
            
            # Store transitions
            for env_idx in range(self.n_envs):
                batch_obs.append(observations[env_idx].copy())
                batch_actions.append(actions[env_idx])
                batch_rewards.append(total_rewards[env_idx])
                batch_values.append(values[env_idx])
                batch_log_probs.append(log_probs[env_idx])
                batch_dones.append(dones[env_idx])
                batch_infos.append(infos[env_idx])
            
            # Archive management (sample from environments to avoid overwhelming archive)
            if self.archive is not None and step_idx % 10 == 0:  # Sample every 10 steps
                savestates = self.env.get_savestates()
                for env_idx in range(self.n_envs):
                    was_novel = self.archive.add_if_novel(
                        next_observations[env_idx], 
                        savestates[env_idx], 
                        self.global_step + step_idx * self.n_envs + env_idx
                    )
                    if was_novel:
                        self.logger.debug(f"Novel state discovered in env {env_idx}")
            
            # Handle environment resets
            reset_indices = []
            for env_idx in range(self.n_envs):
                self.env_step_counts[env_idx] += 1
                
                if dones[env_idx]:
                    # Track episode completion
                    episode_reward = sum(batch_rewards[-self.env_step_counts[env_idx]:])
                    episode_length = self.env_step_counts[env_idx]
                    
                    self.env_episode_rewards[env_idx].append(episode_reward)
                    self.env_episode_lengths[env_idx].append(episode_length)
                    self.env_step_counts[env_idx] = 0
                    
                    reset_indices.append(env_idx)
                    self.episode_count += 1
                    
                    # Always log episode completions to track resets
                    print(f"[EPISODE] Completed in env {env_idx}: reward={episode_reward:.3f}, length={episode_length}, total_episodes={self.episode_count}")
                    if len(reset_indices) <= 2:  # Avoid debug spam
                        self.logger.debug(f"Episode completed in env {env_idx}: reward={episode_reward:.3f}, length={episode_length}")
            
            # Reset completed environments
            if reset_indices:
                archive_size = len(self.archive.cells) if self.archive else 0
                p_frontier = self.config.archive.p_frontier if self.archive else 0.0
                random_roll = np.random.random()
                
                print(f"[RESET] DECISION: {len(reset_indices)} episodes ended")
                print(f"   Archive size: {archive_size} (need >100)")
                print(f"   Frontier probability: {p_frontier}")
                print(f"   Random roll: {random_roll:.3f}")
                
                # Use frontier sampling if archive has sufficient states
                if (self.archive is not None and 
                    len(self.archive.cells) > 100 and 
                    random_roll < self.config.archive.p_frontier):
                    
                    print(f"[FRONTIER] SAMPLING ACTIVATED (roll {random_roll:.3f} < {p_frontier})")
                    
                    # Sample frontier states for all reset environments
                    frontier_states = []
                    for env_idx in reset_indices:
                        frontier_cell = self.archive.sample_frontier()
                        if frontier_cell:
                            frontier_states.append(frontier_cell.savestate)
                            print(f"   Using frontier state for env {env_idx} (visit count: {frontier_cell.visit_count})")
                        else:
                            frontier_states.append(None)
                            print(f"   No frontier state available for env {env_idx}, using normal reset")
                    
                    # Reset all environments with their respective states (frontier or None)
                    self.env.reset(env_indices=reset_indices, from_states=frontier_states)
                    
                    frontier_count = sum(1 for s in frontier_states if s is not None)
                    print(f"[SUMMARY] {frontier_count}/{len(reset_indices)} from frontier, {len(reset_indices) - frontier_count} from game start")
                    self.logger.debug(f"Reset {len(reset_indices)} environments: {frontier_count} from frontier, {len(reset_indices) - frontier_count} from game start")
                else:
                    if self.archive is None:
                        reason = "no archive"
                    elif len(self.archive.cells) <= 100:
                        reason = f"archive too small ({archive_size} <= 100)"
                    else:
                        reason = f"random roll failed ({random_roll:.3f} >= {p_frontier})"
                    
                    print(f"[FRONTIER] SAMPLING SKIPPED: {reason}")
                    print(f"[RESET] Using normal reset from game_start.state for all {len(reset_indices)} environments")
                    
                    # Regular reset without frontier sampling
                    self.env.reset(env_indices=reset_indices)
                    self.logger.debug(f"Reset {len(reset_indices)} environments from game start")
            
            # Update observations for next step
            observations = next_observations
            self.global_step += self.n_envs
        
        # Convert to arrays
        return VectorTransitionBatch(
            obs=np.array(batch_obs),
            actions=np.array(batch_actions),
            rewards=np.array(batch_rewards),
            values=np.array(batch_values),
            log_probs=np.array(batch_log_probs),
            dones=np.array(batch_dones),
            infos=batch_infos
        )
    
    def train_step(self, batch: VectorTransitionBatch) -> Dict[str, float]:
        """
        Perform one training step on the batch.
        
        Args:
            batch: Batch of transitions
            
        Returns:
            Training metrics
        """
        if self.actor_critic is None or self.rnd is None:
            raise RuntimeError("Networks must be initialized")
        
        # Compute advantages using GAE
        advantages, returns = compute_gae(
            rewards=batch.rewards.tolist(),
            values=batch.values.tolist(),
            dones=batch.dones.tolist(),
            gamma=self.config.algo.gamma,
            gae_lambda=self.config.algo.gae_lambda
        )
        
        # Update actor-critic
        ac_metrics = self.actor_critic.update(
            obs_batch=batch.obs,
            act_batch=batch.actions,
            advantages=advantages,
            returns=returns,
            lr_policy=self.config.algo.lr_policy,
            lr_value=self.config.algo.lr_value,
            entropy_coeff=self.config.algo.entropy_coeff,
            value_coeff=self.config.algo.value_coeff,
            grad_clip=self.config.algo.grad_clip
        )
        
        # Extract metrics
        policy_loss = ac_metrics['policy_loss']
        value_loss = ac_metrics['value_loss']
        entropy = ac_metrics['entropy']
        policy_grad_norm = ac_metrics['policy_grad_norm']
        value_grad_norm = ac_metrics['value_grad_norm']
        
        # Update RND with global step for reset tracking
        rnd_metrics = self.rnd.update(batch.obs, self.config.rnd.lr, self.global_step)
        rnd_loss = rnd_metrics['rnd_loss']
        
        # Calculate metrics
        intrinsic_reward_mean = np.mean([self.rnd.intrinsic_reward(obs) for obs in batch.obs[:100]])  # Sample for efficiency
        
        total_loss = policy_loss + value_loss + rnd_loss
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'total_loss': total_loss,
            'rnd_loss': rnd_loss,
            'intrinsic_reward_mean': intrinsic_reward_mean,
            'policy_grad_norm': policy_grad_norm,
            'value_grad_norm': value_grad_norm
        }
    
    def log_metrics(self, epoch: int, train_metrics: Dict[str, float], batch: VectorTransitionBatch) -> None:
        """Log training metrics."""
        # Calculate episode statistics
        all_rewards = [r for env_rewards in self.env_episode_rewards for r in env_rewards[-10:]]  # Last 10 episodes per env
        all_lengths = [l for env_lengths in self.env_episode_lengths for l in env_lengths[-10:]]
        
        episode_reward_mean = np.mean(all_rewards) if all_rewards else 0.0
        episode_length_mean = np.mean(all_lengths) if all_lengths else 0.0
        
        # Calculate total FPS from infos
        total_fps = sum(info.get('fps', 0) for info in batch.infos[-self.n_envs:])  # Last step from each env
        avg_fps_per_env = total_fps / self.n_envs
        
        # Archive statistics
        archive_size = len(self.archive.cells) if self.archive else 0
        archive_novel_rate = self.archive.cells_added / max(self.archive.total_steps, 1) if self.archive else 0
        
        # Cells per hour calculation
        elapsed_hours = (time.time() - getattr(self, '_start_time', time.time())) / 3600
        cells_per_hour = archive_size / max(elapsed_hours, 0.01)
        
        # Log to CSV
        metrics = {
            'epoch': epoch,
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'total_fps': total_fps,
            'avg_fps_per_env': avg_fps_per_env,
            'episode_reward_mean': episode_reward_mean,
            'episode_length_mean': episode_length_mean,
            'archive_size': archive_size,
            'archive_novel_rate': archive_novel_rate,
            'cells_per_hour': cells_per_hour,
            'n_envs': self.n_envs,
            'visual_env_idx': self.visual_env_idx,
            **train_metrics
        }
        
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([metrics[key] for key in [
                "epoch", "global_step", "episode_count", "total_fps", "avg_fps_per_env",
                "episode_reward_mean", "episode_length_mean",
                "policy_loss", "value_loss", "entropy", "total_loss",
                "rnd_loss", "intrinsic_reward_mean",
                "archive_size", "archive_novel_rate", "cells_per_hour",
                "policy_grad_norm", "value_grad_norm",
                "n_envs", "visual_env_idx"
            ]])
        
        self.metrics_history.append(metrics)
        
        # Generate dashboard JSON periodically (every 10 epochs or if logging)
        if self.track_events and (epoch % 10 == 0 or epoch % self.config.train.log_every == 0):
            try:
                from ..tracking.dashboard_generator import generate_dashboard_data
                generate_dashboard_data(self.event_log_dir, self.n_envs)
            except Exception as e:
                self.logger.debug(f"Failed to generate dashboard data: {e}")
        
        # Console logging
        if epoch % self.config.train.log_every == 0:
            print(f"Epoch {epoch:5d} | Step {self.global_step:8d} | "
                  f"Reward: {episode_reward_mean:7.3f} | Length: {episode_length_mean:5.1f} | "
                  f"FPS: {total_fps:6.1f} | Archive: {archive_size:5d} | "
                  f"Loss: {train_metrics['total_loss']:.4f}")
    
    def save_checkpoint(self, epoch: int, path: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'actor_critic_state': self.actor_critic.save_state() if self.actor_critic else None,
            'rnd_state': self.rnd.save_state() if self.rnd else None,
            'archive_state': self.archive.save_state() if self.archive else None,
            'config': self.config,
            'n_envs': self.n_envs,
            'visual_env_idx': self.visual_env_idx
        }
        
        np.savez_compressed(path, **checkpoint)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str) -> int:
        """Load training checkpoint."""
        checkpoint = np.load(path, allow_pickle=True)
        
        self.global_step = int(checkpoint['global_step'])
        self.episode_count = int(checkpoint['episode_count'])
        
        if self.actor_critic and 'actor_critic_state' in checkpoint:
            self.actor_critic.load_state(checkpoint['actor_critic_state'].item())
        
        if self.rnd and 'rnd_state' in checkpoint:
            self.rnd.load_state(checkpoint['rnd_state'].item())
        
        if self.archive and 'archive_state' in checkpoint:
            self.archive.load_state(checkpoint['archive_state'].item())
        
        epoch = int(checkpoint['epoch'])
        print(f"Checkpoint loaded: {path} (epoch {epoch})")
        return epoch
    
    def train(self, rom_path: str, resume_path: Optional[str] = None) -> None:
        """
        Main training loop.
        
        Args:
            rom_path: Path to Game Boy ROM file
            resume_path: Path to checkpoint to resume from
        """
        print("Starting vectorized training...")
        
        # Setup environment and networks
        self.setup_environment(rom_path)
        
        # Resume from checkpoint if provided
        start_epoch = 0
        if resume_path:
            start_epoch = self.load_checkpoint(resume_path) + 1
        
        # Training loop
        self._start_time = time.time()
        
        try:
            for epoch in range(start_epoch, self.config.train.epochs):
                epoch_start = time.time()
                
                # Collect rollout
                batch = self.rollout()
                
                # Train on batch
                train_metrics = self.train_step(batch)
                
                # Log metrics
                self.log_metrics(epoch, train_metrics, batch)
                
                # Save checkpoint
                if epoch % self.config.train.ckpt_every == 0:
                    ckpt_path = os.path.join(self.config.train.save_dir, f"vector_checkpoint_{epoch:06d}.npz")
                    self.save_checkpoint(epoch, ckpt_path)
                
                epoch_time = time.time() - epoch_start
                if epoch % self.config.train.log_every == 0:
                    print(f"  Epoch time: {epoch_time:.2f}s")
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        finally:
            # Clean up
            if self.env:
                self.env.close()
    
    def close(self) -> None:
        """Clean up resources."""
        if self.env:
            self.env.close()