"""Main training loop for NeuralQuest agent."""

import numpy as np
import os
import csv
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from ..envs.pyboy_env import Env
from ..algo.a2c_numpy import ActorCritic, compute_gae
from ..nets.rnd_numpy import RND
from ..explore.archive import Archive
from .config import Config


@dataclass
class Transition:
    """Single environment transition."""
    obs: np.ndarray
    action: int
    reward: float
    value: float
    log_prob: float
    done: bool
    info: Dict[str, Any]


class Trainer:
    """
    Main trainer for NeuralQuest agent.
    
    Coordinates environment interaction, policy learning, curiosity-driven
    exploration, and archive-based state management.
    """
    
    def __init__(self, config: Config):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration
        """
        self.config = config
        config.ensure_dirs()
        
        # Initialize environment (will be set up with ROM path later)
        self.env: Optional[Env] = None
        
        # Initialize networks (will be sized after env setup)
        self.actor_critic: Optional[ActorCritic] = None
        self.rnd: Optional[RND] = None
        self.archive: Optional[Archive] = None
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.total_reward = 0.0
        
        # Metrics tracking
        self.metrics_history: List[Dict[str, Any]] = []
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        
        # Setup logging
        self.setup_logging()
    
    def setup_environment(self, rom_path: str) -> None:
        """
        Setup environment and initialize networks based on observation space.
        
        Args:
            rom_path: Path to Game Boy ROM file
        """
        print(f"Setting up environment with ROM: {rom_path}")
        
        # Initialize environment
        self.env = Env(
            rom_path=rom_path,
            frame_skip=self.config.env.frame_skip,
            sticky_p=self.config.env.sticky_p,
            max_episode_steps=self.config.env.max_episode_steps,
            deterministic=self.config.env.deterministic,
            headless=self.config.env.headless,
            seed=self.config.env.seed
        )
        
        # Get observation dimension
        obs_dim = self.env.obs_dim
        print(f"Observation dimension: {obs_dim}")
        
        # Initialize networks
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
        
        self.archive = Archive(
            input_dim=obs_dim,
            capacity=self.config.archive.capacity,
            novel_lru=self.config.archive.novel_lru,
            hamming_threshold=self.config.archive.hamming_threshold,
            hash_bits=self.config.archive.hash_bits,
            projection_dim=self.config.archive.projection_dim,
            seed=self.config.env.seed
        )
        
        print("Networks initialized successfully")
    
    def setup_logging(self) -> None:
        """Setup CSV logging for metrics."""
        self.metrics_file = os.path.join(self.config.train.log_dir, "metrics.csv")
        
        # Create CSV header
        headers = [
            "epoch", "global_step", "episode_count", "fps",
            "episode_reward_mean", "episode_length_mean",
            "policy_loss", "value_loss", "entropy", "total_loss",
            "rnd_loss", "intrinsic_reward_mean",
            "archive_size", "archive_novel_rate", "cells_per_hour",
            "policy_grad_norm", "value_grad_norm"
        ]
        
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def rollout(self) -> List[Transition]:
        """
        Collect a rollout of environment interactions.
        
        Returns:
            List of transitions
        """
        if self.env is None or self.actor_critic is None:
            raise RuntimeError("Environment and networks must be initialized")
        
        transitions = []
        # Only reset if this is the very first rollout
        if self.global_step == 0:
            obs = self.env.reset()
            print(f"First rollout - reset environment")
        else:
            # Continue from where we left off - get current observation
            obs = self.env._get_observation()
            print(f"Continuing rollout from global step {self.global_step}")
        
        print(f"Starting rollout: batch_horizon={self.config.algo.batch_horizon}, global_step={self.global_step}", flush=True)
        
        for step_idx in range(self.config.algo.batch_horizon):
            # Get action from policy
            action, log_prob, value, aux_info = self.actor_critic.act(obs)
            
            # Step environment
            next_obs, env_reward, done, info = self.env.step(action)
            
            # Debug episode progression
            if step_idx % 500 == 0 or done:
                print(f"Step {step_idx} (global {self.global_step}): episode_step={info['episode_step']}, done={done}, max_steps={self.config.env.max_episode_steps}")
            
            # Compute intrinsic reward
            intrinsic_reward = self.rnd.intrinsic_reward(next_obs)
            total_reward = self.config.rnd.beta * intrinsic_reward
            
            # Store transition
            transition = Transition(
                obs=obs.copy(),
                action=action,
                reward=total_reward,
                value=value,
                log_prob=log_prob,
                done=done,
                info=info
            )
            transitions.append(transition)
            
            # Archive management
            if self.archive is not None:
                savestate = self.env.save_state()
                was_novel = self.archive.add_if_novel(next_obs, savestate, self.global_step)
                
                if was_novel and self.global_step % 1000 == 0:
                    print(f"Novel state discovered at step {self.global_step}")
            
            # Update state
            obs = next_obs
            self.global_step += 1
            
            # Episode end handling
            if done:
                # Record episode completion BEFORE reset
                episode_length = info["episode_step"]
                self.episode_lengths.append(episode_length)
                self.episode_count += 1
                
                # Calculate episode reward from the transitions that belong to this episode
                episode_start_idx = max(0, len(transitions) - episode_length)
                episode_reward = sum(t.reward for t in transitions[episode_start_idx:])
                self.episode_rewards.append(episode_reward)
                
                print(f"Episode {self.episode_count} completed: {episode_length} steps, reward: {episode_reward:.3f}", flush=True)
                print(f"Frontier probability: {self.config.archive.p_frontier}", flush=True)
                
                # Maybe reset from frontier
                try:
                    rand_val = np.random.random()
                    print(f"Frontier sampling: rand={rand_val:.3f}, p_frontier={self.config.archive.p_frontier}", flush=True)
                    
                    if (self.archive is not None and rand_val < self.config.archive.p_frontier):
                        print(f"Attempting frontier sampling...", flush=True)
                        frontier_cell = self.archive.sample_frontier()
                        if frontier_cell is not None:
                            print(f"SUCCESS: Reset from frontier cell {frontier_cell.cell_id} (step {frontier_cell.first_seen_step})", flush=True)
                            # Fix: Don't call load_state() AND reset(from_state=...) - that loads twice!
                            obs = self.env.reset(from_state=frontier_cell.savestate)
                        else:
                            print(f"FAILED: No frontier cell available, doing regular reset", flush=True)
                            obs = self.env.reset()
                    else:
                        print(f"Regular reset (no frontier sampling)", flush=True)
                        obs = self.env.reset()
                except Exception as e:
                    print(f"ERROR in frontier sampling: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    obs = self.env.reset()
        
        return transitions
    
    def update_networks(self, transitions: List[Transition]) -> Dict[str, float]:
        """
        Update actor-critic and RND networks.
        
        Args:
            transitions: List of collected transitions
            
        Returns:
            Training statistics
        """
        if not transitions:
            return {}
        
        # Extract data from transitions
        obs_batch = np.array([t.obs for t in transitions])
        act_batch = np.array([t.action for t in transitions])
        rewards = [t.reward for t in transitions]
        values = [t.value for t in transitions]
        dones = [t.done for t in transitions]
        
        # Compute final value for GAE
        if dones[-1]:
            last_value = 0.0
        else:
            # Use current policy to estimate final state value
            final_obs = transitions[-1].obs
            _, _, last_value, _ = self.actor_critic.act(final_obs)
        
        # Compute advantages and returns using GAE
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=self.config.algo.gamma,
            gae_lambda=self.config.algo.gae_lambda,
            last_value=last_value
        )
        
        # Update actor-critic
        ac_stats = self.actor_critic.update(
            obs_batch=obs_batch,
            act_batch=act_batch,
            advantages=advantages,
            returns=returns,
            lr_policy=self.config.algo.lr_policy,
            lr_value=self.config.algo.lr_value,
            entropy_coeff=self.config.algo.entropy_coeff,
            value_coeff=self.config.algo.value_coeff,
            grad_clip=self.config.algo.grad_clip
        )
        
        # Update RND predictor
        rnd_stats = self.rnd.update(obs_batch, self.config.rnd.lr)
        
        # Combine statistics
        stats = {**ac_stats, **rnd_stats}
        stats['intrinsic_reward_mean'] = np.mean([self.rnd.intrinsic_reward(obs) for obs in obs_batch[:10]])
        
        return stats
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch.
        
        Returns:
            Epoch statistics
        """
        start_time = time.time()
        
        # Collect rollout
        transitions = self.rollout()
        
        # Update networks
        update_stats = self.update_networks(transitions)
        
        # Compute metrics
        epoch_time = time.time() - start_time
        fps = len(transitions) / epoch_time if epoch_time > 0 else 0
        
        # Archive statistics
        archive_stats = self.archive.stats() if self.archive else {}
        
        # Episode statistics
        recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else [0]
        recent_lengths = self.episode_lengths[-100:] if self.episode_lengths else [0]
        
        metrics = {
            'fps': fps,
            'episode_reward_mean': np.mean(recent_rewards),
            'episode_length_mean': np.mean(recent_lengths),
            'archive_size': archive_stats.get('size', 0),
            'archive_novel_rate': archive_stats.get('cells_added', 0) / max(self.global_step, 1) * 1000,
            'cells_per_hour': archive_stats.get('cells_added', 0) / max(epoch_time / 3600, 1e-6),
            **update_stats
        }
        
        return metrics
    
    def train(self, rom_path: str, total_epochs: Optional[int] = None) -> None:
        """
        Main training loop.
        
        Args:
            rom_path: Path to Game Boy ROM file
            total_epochs: Total epochs to train (uses config if None)
        """
        # Setup environment and networks
        self.setup_environment(rom_path)
        
        if total_epochs is None:
            total_epochs = self.config.train.epochs
        
        print(f"Starting training for {total_epochs} epochs")
        print(f"Configuration: {self.config}")
        
        try:
            for epoch in range(total_epochs):
                # Train one epoch
                metrics = self.train_epoch()
                metrics['epoch'] = epoch
                metrics['global_step'] = self.global_step
                metrics['episode_count'] = self.episode_count
                
                # Log metrics
                self.log_metrics(metrics)
                
                # Print progress
                if epoch % self.config.train.log_every == 0:
                    self.print_progress(epoch, metrics)
                
                # Save checkpoint
                if epoch % self.config.train.ckpt_every == 0:
                    self.save_checkpoint(epoch)
                
                # Evaluation
                if epoch % self.config.train.eval_every == 0:
                    self.evaluate()
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self.save_checkpoint(epoch, prefix="interrupted")
        
        finally:
            if self.env:
                self.env.close()
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to CSV file."""
        self.metrics_history.append(metrics)
        
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                metrics.get('epoch', 0),
                metrics.get('global_step', 0),
                metrics.get('episode_count', 0),
                metrics.get('fps', 0),
                metrics.get('episode_reward_mean', 0),
                metrics.get('episode_length_mean', 0),
                metrics.get('policy_loss', 0),
                metrics.get('value_loss', 0),
                metrics.get('entropy', 0),
                metrics.get('total_loss', 0),
                metrics.get('rnd_loss', 0),
                metrics.get('intrinsic_reward_mean', 0),
                metrics.get('archive_size', 0),
                metrics.get('archive_novel_rate', 0),
                metrics.get('cells_per_hour', 0),
                metrics.get('policy_grad_norm', 0),
                metrics.get('value_grad_norm', 0)
            ]
            writer.writerow(row)
    
    def print_progress(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Print training progress."""
        print(f"Epoch {epoch:6d} | "
              f"Steps {self.global_step:8d} | "
              f"Episodes {self.episode_count:6d} | "
              f"FPS {metrics.get('fps', 0):6.1f} | "
              f"Reward {metrics.get('episode_reward_mean', 0):8.3f} | "
              f"Length {metrics.get('episode_length_mean', 0):6.1f} | "
              f"Archive {metrics.get('archive_size', 0):6d} | "
              f"Entropy {metrics.get('entropy', 0):6.3f}")
    
    def save_checkpoint(self, epoch: int, prefix: str = "epoch") -> None:
        """Save training checkpoint."""
        checkpoint_dir = os.path.join(self.config.train.save_dir, f"{prefix}_{epoch:06d}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save networks
        if self.actor_critic:
            ac_state = {
                'encoder': self.actor_critic.encoder.save_state(),
                'policy_head': self.actor_critic.policy_head.save_state(),
                'value_head': self.actor_critic.value_head.save_state()
            }
            np.savez(os.path.join(checkpoint_dir, "actor_critic.npz"), **ac_state)
        
        if self.rnd:
            rnd_state = self.rnd.save_state()
            np.savez(os.path.join(checkpoint_dir, "rnd.npz"), **rnd_state)
        
        if self.archive:
            archive_state = self.archive.save_state()
            np.savez(os.path.join(checkpoint_dir, "archive.npz"), **archive_state)
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'config': self.config.__dict__
        }
        np.savez(os.path.join(checkpoint_dir, "training_state.npz"), **training_state)
        
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load training checkpoint."""
        print(f"Loading checkpoint from {checkpoint_dir}")
        
        # Load training state
        training_state = np.load(os.path.join(checkpoint_dir, "training_state.npz"), allow_pickle=True)
        self.global_step = int(training_state['global_step'])
        self.episode_count = int(training_state['episode_count'])
        
        # Load networks (requires environment to be set up first)
        if self.actor_critic:
            ac_data = np.load(os.path.join(checkpoint_dir, "actor_critic.npz"), allow_pickle=True)
            self.actor_critic.encoder.load_state(ac_data['encoder'].item())
            self.actor_critic.policy_head.load_state(ac_data['policy_head'].item())
            self.actor_critic.value_head.load_state(ac_data['value_head'].item())
        
        if self.rnd:
            rnd_data = np.load(os.path.join(checkpoint_dir, "rnd.npz"), allow_pickle=True)
            # Convert numpy arrays back to the expected format
            rnd_state = {key: value.item() if isinstance(value, np.ndarray) and value.shape == () else value 
                        for key, value in rnd_data.items()}
            self.rnd.load_state(rnd_state)
        
        if self.archive:
            archive_data = np.load(os.path.join(checkpoint_dir, "archive.npz"), allow_pickle=True)
            archive_state = {key: value.item() if isinstance(value, np.ndarray) and value.shape == () else value 
                           for key, value in archive_data.items()}
            self.archive.load_state(archive_state)
        
        print("Checkpoint loaded successfully")
    
    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """
        Evaluate current policy.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation metrics
        """
        if self.env is None or self.actor_critic is None:
            return {}
        
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Greedy action selection (no exploration)
                action, _, _, _ = self.actor_critic.act(obs)
                obs, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if episode_length >= self.config.env.max_episode_steps:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        eval_stats = {
            'eval_reward_mean': np.mean(eval_rewards),
            'eval_reward_std': np.std(eval_rewards),
            'eval_length_mean': np.mean(eval_lengths),
            'eval_length_std': np.std(eval_lengths)
        }
        
        print(f"Evaluation: Reward {eval_stats['eval_reward_mean']:.3f} ± {eval_stats['eval_reward_std']:.3f}, "
              f"Length {eval_stats['eval_length_mean']:.1f} ± {eval_stats['eval_length_std']:.1f}")
        
        return eval_stats