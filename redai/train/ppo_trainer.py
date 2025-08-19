"""PPO trainer for Pokemon Red speedrunning with checkpoints and curriculum learning."""

import numpy as np
import time
import csv
import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pickle
import gzip
from dataclasses import dataclass

from ..algo.ppo_numpy import PPOActorCritic
from ..nets.rnd_numpy import RND
from ..envs.pyboy_env import Env
from ..envs.checkpoint_system import CheckpointManager
# from ..explore.archive import Archive  # Removed - not needed for speedrunning


@dataclass
class PPOConfig:
    """PPO training configuration."""
    # PPO specific parameters
    clip_ratio: float = 0.2
    ppo_epochs: int = 4
    minibatch_size: int = 256
    target_kl: float = 0.01
    
    # Standard RL parameters
    gamma: float = 0.999
    gae_lambda: float = 0.95
    lr_policy: float = 3e-4
    lr_value: float = 3e-4
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    grad_clip: float = 0.5
    
    # Training parameters  
    batch_horizon: int = 10000  # Learn frequently (~30 seconds of experience)
    max_episode_steps: int = 10000000  # Still allow very long episodes
    epochs: int = 10000
    
    # RND parameters
    use_rnd: bool = True
    rnd_lr: float = 1e-3
    rnd_beta: float = 0.1
    
    # Archive parameters
    use_archive: bool = False  # Disable for speedrunning focus
    
    # Checkpoint and curriculum
    use_checkpoints: bool = True
    curriculum_stage: str = "adaptive"  # "easy", "normal", "hard", "mixed", "adaptive"
    checkpoint_every: int = 50  # Save checkpoint every N epochs
    
    # Logging
    log_every: int = 10
    eval_episodes: int = 5


class PPOTrainer:
    """
    PPO trainer optimized for Pokemon Red speedrunning.
    
    Features:
    - PPO algorithm with clipped objectives
    - Checkpoint-based curriculum learning
    - Intrinsic motivation via RND
    - Progress tracking and visualization
    """
    
    def __init__(
        self,
        env: Env,
        config: PPOConfig,
        save_dir: str = "pokemon_ppo_checkpoints",
        log_dir: str = "pokemon_ppo_logs"
    ):
        """
        Initialize PPO trainer.
        
        Args:
            env: Pokemon environment
            config: Training configuration
            save_dir: Directory to save model checkpoints
            log_dir: Directory to save training logs
        """
        self.env = env
        self.config = config
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.save_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Get observation dimension
        dummy_obs = env.reset()
        obs_dim = len(dummy_obs)
        
        # Initialize PPO actor-critic
        self.agent = PPOActorCritic(
            input_dim=obs_dim,
            n_actions=len(env.ACTIONS),
            hidden_dim=512,
            seed=42
        )
        
        # Initialize RND if enabled
        self.rnd = None
        if config.use_rnd:
            self.rnd = RND(
                input_dim=obs_dim,
                hidden_dim=256,
                seed=123
            )
        
        # Archive system removed for speedrunning focus
        
        # Initialize checkpoint system
        self.checkpoint_manager = None
        if config.use_checkpoints:
            self.checkpoint_manager = CheckpointManager("pokemon_speedrun_checkpoints")
        
        # Training state
        self.epoch = 0
        self.total_steps = 0
        self.best_completion_time = float('inf')
        self.best_completion_episode = -1
        
        # Stuck detection for speedrunning
        self.progress_history = []  # Track recent progress
        self.stuck_threshold = 2000  # Steps without meaningful progress (about 2-3 minutes)
        self.last_significant_progress = 0
        self.current_episode_start_step = 0
        self.min_progress_time = 500  # Minimum steps before checking for stuck
        
        # Initialize CSV logger
        self.csv_path = self.log_dir / "training_log.csv"
        self._init_csv_logger()
        
        print("PPO Trainer initialized for Pokemon Red speedrunning")
        print(f"   - Observation dim: {obs_dim}")
        print(f"   - Action space: {len(env.ACTIONS)} actions")
        print(f"   - RND enabled: {config.use_rnd}")
        print(f"   - Checkpoints enabled: {config.use_checkpoints}")
        print(f"   - Curriculum stage: {config.curriculum_stage}")
    
    def _init_csv_logger(self):
        """Initialize CSV logging."""
        headers = [
            'epoch', 'total_steps', 'episode_reward', 'episode_length',
            'badges_earned', 'pokemon_count', 'game_completed', 'completion_time',
            'policy_loss', 'value_loss', 'entropy', 'kl_divergence',
            'explained_variance', 'clip_fraction', 'fps'
        ]
        
        if self.config.use_rnd:
            headers.extend(['rnd_loss', 'intrinsic_reward'])
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def _log_to_csv(self, data: Dict[str, Any]):
        """Log training data to CSV."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                data.get('epoch', ''),
                data.get('total_steps', ''),
                data.get('episode_reward', ''),
                data.get('episode_length', ''),
                data.get('badges_earned', ''),
                data.get('pokemon_count', ''),
                data.get('game_completed', ''),
                data.get('completion_time', ''),
                data.get('policy_loss', ''),
                data.get('value_loss', ''),
                data.get('entropy', ''),
                data.get('kl_divergence', ''),
                data.get('explained_variance', ''),
                data.get('clip_fraction', ''),
                data.get('fps', '')
            ]
            
            if self.config.use_rnd:
                row.extend([
                    data.get('rnd_loss', ''),
                    data.get('intrinsic_reward', '')
                ])
            
            writer.writerow(row)
    
    def _select_start_state(self) -> Optional[bytes]:
        """Always start from beginning for speedrunning - no checkpoints."""
        # For true speedrunning, we always start from the beginning
        # The agent learns complete end-to-end runs getting faster each time
        return None
    
    def _compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        rewards = np.array(rewards)
        values = np.array(values + [next_value])
        dones = np.array(dones)
        
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                delta = rewards[i] - values[i]
                gae = delta
            else:
                delta = rewards[i] + self.config.gamma * values[i + 1] - values[i]
                gae = delta + self.config.gamma * self.config.gae_lambda * gae
            
            advantages[i] = gae
        
        returns = advantages + values[:-1]
        return advantages, returns
    
    def _collect_rollouts(self) -> Dict[str, List]:
        """Collect rollout data for PPO training."""
        observations = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        infos = []
        
        # For continuous speedrunning, don't reset environment between epochs
        # Only reset on first epoch or when episode naturally ends
        if self.epoch == 0 or not hasattr(self, '_current_obs'):
            start_state = self._select_start_state()
            obs = self.env.reset(from_state=start_state)
            self._current_obs = obs
            print(f"Environment reset for epoch {self.epoch}")
        else:
            obs = self._current_obs
            print(f"Continuing from current game state for epoch {self.epoch}")
        
        episode_rewards = []
        episode_lengths = []
        episode_info = {}
        
        steps_collected = 0
        episode_steps = 0
        episode_reward = 0.0
        episode_progress = 0  # Track badges for this episode
        
        print(f"Starting rollout collection with 10M step episodes for full game completion attempts.")
        
        start_time = time.time()
        
        while steps_collected < self.config.batch_horizon:
            # Print new episode start
            if episode_steps == 0:
                print(f"New episode started - aiming for Pokemon Red speedrun completion!")
            
            # Get action from policy
            action, log_prob, value, aux_info = self.agent.act(obs)
            
            # Take environment step
            next_obs, reward, done, info = self.env.step(action)
            
            # Check for unexpected immediate episode end
            if done and episode_steps == 0:
                print(f"WARNING: Episode ended immediately! done={done}")
            
            # Add RND intrinsic reward if enabled
            if self.config.use_rnd and self.rnd is not None:
                intrinsic_reward = self.rnd.intrinsic_reward(obs)
                reward += self.config.rnd_beta * intrinsic_reward
                info['intrinsic_reward'] = intrinsic_reward
            
            # Store rollout data
            observations.append(obs.copy())
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done)
            infos.append(info.copy())
            
            # Track significant progress for stuck detection
            badges_now = info.get('badges_count', 0)
            pokemon_now = info.get('pokemon_count', 0)
            maps_visited = info.get('maps_visited', 0)
            
            # Define what counts as meaningful progress
            made_progress = False
            
            # Print progress tracking info every 500 steps (less verbose)
            if episode_steps % 500 == 0:
                print(f"Progress: Step {episode_steps} | Badges: {badges_now} | Pokemon: {pokemon_now} | Maps: {maps_visited}")
            
            if badges_now > episode_progress:
                self.last_significant_progress = episode_steps
                print(f"PROGRESS: Badge #{badges_now} earned at step {episode_steps}!")
                made_progress = True
            elif pokemon_now > 0 and episode_steps < 1000:  # Early game Pokemon catching
                self.last_significant_progress = episode_steps
                made_progress = True
            elif maps_visited > 0:  # Any map exploration counts as progress
                self.last_significant_progress = episode_steps
                made_progress = True
            
            # Update episode progress tracking
            episode_progress = badges_now
            
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            steps_collected += 1
            
            # Handle episode termination or stuck detection
            episode_progress = info.get('badges_count', 0)
            steps_since_progress = episode_steps - self.last_significant_progress
            
            # Check if agent is stuck (no progress for too long)
            # Only consider stuck if:
            # 1. Episode is long enough (> min_progress_time)
            # 2. No badges earned AND no Pokemon caught AND no meaningful exploration
            # 3. Been stuck for longer than threshold
            maps_visited = info.get('maps_visited', 0)
            
            # Be more lenient - only consider stuck if we have clear evidence of being stuck
            # and we've given enough time for the startup sequence
            is_stuck = (episode_steps > max(self.min_progress_time, 1000) and  # At least 1000 steps
                       episode_progress == 0 and 
                       info.get('pokemon_count', 0) == 0 and
                       maps_visited == 0 and
                       steps_since_progress > self.stuck_threshold and
                       episode_steps > 0)  # Ensure episode actually started
            
            if done or is_stuck:
                if is_stuck:
                    print(f"STUCK DETECTION: Restarting after {episode_steps} steps (no progress for {steps_since_progress} steps)")
                    print(f"  Progress data: badges={episode_progress}, pokemon={info.get('pokemon_count', 0)}, maps={maps_visited}")
                    done = True  # Force episode end
                
                # Always process episode end (even if 0 steps - but log it)
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_steps)
                episode_info = info.copy()
                
                # Log unusual episode lengths
                if episode_steps == 0:
                    print(f"WARNING: Episode ended with 0 steps! done={done}, is_stuck={is_stuck}")
                    print(f"  steps_collected={steps_collected}, env_step_count={info.get('episode_step', '?')}")
                elif episode_steps < 50:
                    print(f"WARNING: Very short episode: {episode_steps} steps")
                
                # Track progress for stuck detection
                self.progress_history.append({
                    'steps': episode_steps,
                    'badges': episode_progress,
                    'reward': episode_reward,
                    'completed': info.get('game_completed', False)
                })
                
                # Keep only recent history
                if len(self.progress_history) > 10:
                    self.progress_history.pop(0)
                
                # Check if game was completed
                if info.get('game_completed', False):
                    completion_time = info.get('completion_time', -1)
                    if completion_time < self.best_completion_time and completion_time > 0:
                        self.best_completion_time = completion_time
                        self.best_completion_episode = self.epoch
                        print(f"NEW BEST COMPLETION TIME: {completion_time} steps!")
                        
                        # Save this achievement
                        self._save_checkpoint(f"best_completion_epoch_{self.epoch}")
                
                # Reset for next episode only when truly necessary (stuck or completed)
                if is_stuck or info.get('game_completed', False):
                    print(f"Resetting environment for new episode (step_collected={steps_collected})")
                    obs = self.env.reset()  # Reset when stuck or game completed
                    self._current_obs = obs  # Update stored observation
                else:
                    # Natural episode end (max steps) - don't reset, continue from current state
                    print(f"Episode ended naturally at {episode_steps} steps, continuing...")
                
                episode_reward = 0.0
                episode_steps = 0
                episode_progress = 0  # Reset badge progress
                self.current_episode_start_step = steps_collected
                self.last_significant_progress = 0
                
                # Continue to next iteration
                continue
        
        # If episode didn't end within batch, record current progress
        if not episode_rewards or len(episode_rewards) == 0:
            print(f"Batch completed without episode end. Recording partial episode: {episode_steps} steps, {episode_reward} reward")
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            episode_info = {
                'badges_count': 0,  # Default values if info not available
                'pokemon_count': 0,
                'game_completed': False,
                'completion_time': -1,
                'episode_step': episode_steps
            }
        
        # Compute final value for GAE
        _, _, final_value, _ = self.agent.act(obs)
        
        # Compute advantages and returns
        advantages, returns = self._compute_gae(rewards, values, dones, final_value)
        
        elapsed_time = time.time() - start_time
        fps = steps_collected / elapsed_time
        
        # Save current observation for next epoch (continuous gameplay)
        self._current_obs = obs.copy()
        
        print(f"Rollout collection complete: {steps_collected} steps, {len(episode_rewards)} episodes")
        if episode_rewards:
            print(f"  Episode lengths: {episode_lengths}")
            print(f"  Episode rewards: {[f'{r:.1f}' for r in episode_rewards]}")
        
        return {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'values': np.array(values),
            'log_probs': np.array(log_probs),
            'advantages': advantages,
            'returns': returns,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_info': episode_info,
            'fps': fps,
            'steps_collected': steps_collected
        }
    
    def _update_policy(self, rollout_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Update PPO policy using collected rollouts."""
        observations = rollout_data['observations']
        actions = rollout_data['actions']
        old_log_probs = rollout_data['log_probs']
        advantages = rollout_data['advantages']
        returns = rollout_data['returns']
        
        batch_size = len(observations)
        
        # PPO update statistics
        update_stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'kl_divergence': 0.0,
            'explained_variance': 0.0,
            'clip_fraction': 0.0
        }
        
        # Multiple epochs of PPO updates
        for epoch in range(self.config.ppo_epochs):
            # Create minibatches
            indices = np.random.permutation(batch_size)
            
            for start_idx in range(0, batch_size, self.config.minibatch_size):
                end_idx = min(start_idx + self.config.minibatch_size, batch_size)
                mb_indices = indices[start_idx:end_idx]
                
                # Extract minibatch
                mb_obs = observations[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # PPO update
                stats = self.agent.update(
                    obs_batch=mb_obs,
                    act_batch=mb_actions,
                    advantages=mb_advantages,
                    returns=mb_returns,
                    old_log_probs=mb_old_log_probs,
                    lr_policy=self.config.lr_policy,
                    lr_value=self.config.lr_value,
                    clip_ratio=self.config.clip_ratio,
                    entropy_coeff=self.config.entropy_coeff,
                    value_coeff=self.config.value_coeff,
                    grad_clip=self.config.grad_clip,
                    target_kl=self.config.target_kl
                )
                
                # Accumulate statistics
                for key in update_stats:
                    update_stats[key] += stats[key]
            
            # Check KL divergence for early stopping
            if update_stats['kl_divergence'] > self.config.target_kl * 1.5:
                print(f"Early stopping due to large KL divergence: {update_stats['kl_divergence']:.4f}")
                break
        
        # Average statistics
        num_updates = (epoch + 1) * (batch_size // self.config.minibatch_size)
        for key in update_stats:
            update_stats[key] /= num_updates
        
        # Update old policy for next iteration
        self.agent.update_old_policy()
        
        return update_stats
    
    def _save_checkpoint(self, name: Optional[str] = None):
        """Save training checkpoint."""
        if name is None:
            name = f"epoch_{self.epoch:04d}"
        
        checkpoint_path = self.save_dir / f"{name}.pkl"
        
        checkpoint_data = {
            'epoch': self.epoch,
            'total_steps': self.total_steps,
            'agent_state': self.agent.save_state(),
            'best_completion_time': self.best_completion_time,
            'best_completion_episode': self.best_completion_episode
        }
        
        if self.rnd is not None:
            checkpoint_data['rnd_state'] = self.rnd.save_state()
        
        with gzip.open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        print(f"Saved checkpoint: {name}")
    
    def train(self) -> None:
        """Main training loop."""
        print("Starting PPO training for Pokemon Red speedrunning!")
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            # Collect rollouts
            rollout_data = self._collect_rollouts()
            
            # Update total steps
            self.total_steps += rollout_data['steps_collected']
            
            # Update RND if enabled
            rnd_stats = {}
            if self.config.use_rnd and self.rnd is not None:
                rnd_result = self.rnd.update(rollout_data['observations'], lr=self.config.rnd_lr, global_step=self.total_steps)
                rnd_stats = {'rnd_loss': rnd_result.get('rnd_loss', 0.0)}
            
            # Update policy
            policy_stats = self._update_policy(rollout_data)
            
            # Logging
            if epoch % self.config.log_every == 0:
                episode_info = rollout_data['episode_info']
                log_data = {
                    'epoch': epoch,
                    'total_steps': self.total_steps,
                    'episode_reward': np.mean(rollout_data['episode_rewards']) if rollout_data['episode_rewards'] else 0,
                    'episode_length': np.mean(rollout_data['episode_lengths']) if rollout_data['episode_lengths'] else 0,
                    'badges_earned': episode_info.get('badges_count', 0),
                    'pokemon_count': episode_info.get('pokemon_count', 0),
                    'game_completed': episode_info.get('game_completed', False),
                    'completion_time': episode_info.get('completion_time', -1),
                    'fps': rollout_data['fps'],
                    **policy_stats,
                    **rnd_stats
                }
                
                self._log_to_csv(log_data)
                
                print(f"Epoch {epoch:4d} | "
                      f"Steps: {self.total_steps:7d} | "
                      f"Reward: {log_data['episode_reward']:6.1f} | "
                      f"Length: {log_data['episode_length']:4.0f} | "
                      f"Badges: {log_data['badges_earned']} | "
                      f"Completed: {log_data['game_completed']} | "
                      f"FPS: {log_data['fps']:4.0f}")
            
            # Save checkpoint
            if epoch % self.config.checkpoint_every == 0 and epoch > 0:
                self._save_checkpoint()
        
        print("Training completed!")
        print(f"   Best completion time: {self.best_completion_time} steps (epoch {self.best_completion_episode})")
        self._save_checkpoint("final")