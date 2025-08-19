#!/usr/bin/env python3
"""
Pokemon Red Speedrunning Training Script using PPO

This script trains an agent to speedrun Pokemon Red by:
1. Using PPO algorithm for better sample efficiency
2. Pokemon-specific reward system focused on game completion
3. Checkpoint-based curriculum learning
4. Single environment with visual feedback
5. Progress tracking and monitoring
"""

import argparse
import sys
import os
from pathlib import Path
import tomli
from dataclasses import dataclass, fields

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from redai.envs.pyboy_env import Env
from redai.train.ppo_trainer import PPOTrainer, PPOConfig


def load_config(config_path: str) -> PPOConfig:
    """Load configuration from TOML file and convert to PPOConfig."""
    
    with open(config_path, 'rb') as f:
        toml_data = tomli.load(f)
    
    # Extract PPO-specific config
    ppo_config = toml_data.get('ppo', {})
    rnd_config = toml_data.get('rnd', {})
    checkpoints_config = toml_data.get('checkpoints', {})
    archive_config = toml_data.get('archive', {})
    train_config = toml_data.get('train', {})
    
    # Create PPOConfig with values from TOML
    config_kwargs = {}
    
    # Map TOML values to PPOConfig fields
    field_mapping = {
        'clip_ratio': ppo_config.get('clip_ratio', 0.2),
        'ppo_epochs': ppo_config.get('ppo_epochs', 4),
        'minibatch_size': ppo_config.get('minibatch_size', 256),
        'target_kl': ppo_config.get('target_kl', 0.01),
        'gamma': ppo_config.get('gamma', 0.999),
        'gae_lambda': ppo_config.get('gae_lambda', 0.95),
        'lr_policy': ppo_config.get('lr_policy', 3e-4),
        'lr_value': ppo_config.get('lr_value', 3e-4),
        'entropy_coeff': ppo_config.get('entropy_coeff', 0.01),
        'value_coeff': ppo_config.get('value_coeff', 0.5),
        'grad_clip': ppo_config.get('grad_clip', 0.5),
        'batch_horizon': ppo_config.get('batch_horizon', 4096),
        'max_episode_steps': ppo_config.get('max_episode_steps', 50000),
        'epochs': train_config.get('epochs', 10000),
        'use_rnd': rnd_config.get('use_rnd', True),
        'rnd_lr': rnd_config.get('lr', 1e-3),
        'rnd_beta': rnd_config.get('beta', 0.1),
        'use_archive': archive_config.get('use_archive', False),
        'use_checkpoints': checkpoints_config.get('use_checkpoints', True),
        'curriculum_stage': checkpoints_config.get('curriculum_stage', 'adaptive'),
        'checkpoint_every': train_config.get('checkpoint_every', 50),
        'log_every': train_config.get('log_every', 10),
        'eval_episodes': train_config.get('eval_episodes', 3)
    }
    
    # Only include fields that exist in PPOConfig
    ppo_config_fields = {f.name for f in fields(PPOConfig)}
    for key, value in field_mapping.items():
        if key in ppo_config_fields:
            config_kwargs[key] = value
    
    return PPOConfig(**config_kwargs), toml_data


def create_environment(toml_config: dict, rom_path: str) -> Env:
    """Create Pokemon environment with speedrunning configuration."""
    
    env_config = toml_config.get('env', {})
    
    env = Env(
        rom_path=rom_path,
        frame_skip=env_config.get('frame_skip', 4),
        sticky_p=env_config.get('sticky_p', 0.05),
        frame_stack=4,  # Always use frame stack of 4
        max_episode_steps=env_config.get('max_episode_steps', 50000),
        deterministic=env_config.get('deterministic', True),
        headless=env_config.get('headless', False),
        use_progress_detector=env_config.get('use_progress_detector', False),
        use_speedrun_rewards=env_config.get('use_speedrun_rewards', True),
        seed=env_config.get('seed', 1337)
    )
    
    return env


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Pokemon Red speedrunning agent with PPO")
    parser.add_argument('--config', 
                       default='configs/pokemon_speedrun_ppo.toml',
                       help='Configuration file path')
    parser.add_argument('--rom', 
                       default='roms/pokemon_red.gb',
                       help='Pokemon Red ROM file path')
    parser.add_argument('--save-dir',
                       help='Directory to save checkpoints (overrides config)')
    parser.add_argument('--log-dir', 
                       help='Directory to save logs (overrides config)')
    parser.add_argument('--resume',
                       help='Resume from checkpoint file')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluation, no training')
    parser.add_argument('--test-setup', action='store_true',
                       help='Test setup and exit')
    
    args = parser.parse_args()
    
    # Check ROM file exists
    if not os.path.exists(args.rom):
        print(f"ERROR: ROM file not found: {args.rom}")
        print("Please place Pokemon Red ROM at the specified path")
        print("You must legally own the ROM file")
        sys.exit(1)
    
    # Load configuration
    try:
        ppo_config, toml_config = load_config(args.config)
        print(f"SUCCESS: Loaded configuration from {args.config}")
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        sys.exit(1)
    
    # Override directories if specified
    train_config = toml_config.get('train', {})
    save_dir = args.save_dir or train_config.get('save_dir', 'pokemon_speedrun_checkpoints')
    log_dir = args.log_dir or train_config.get('log_dir', 'pokemon_speedrun_logs')
    
    # Create environment
    try:
        env = create_environment(toml_config, args.rom)
        print("SUCCESS: Pokemon environment created successfully")
        
        # Test environment
        if args.test_setup:
            print("TESTING: Environment setup...")
            obs = env.reset()
            print(f"   Observation shape: {obs.shape}")
            
            for i in range(10):
                action = env.ACTIONS[i % len(env.ACTIONS)]
                obs, reward, done, info = env.step(i % len(env.ACTIONS))
                print(f"   Step {i}: action={action}, reward={reward:.3f}, done={done}")
                if done:
                    obs = env.reset()
            
            print("SUCCESS: Environment test completed successfully")
            return
            
    except Exception as e:
        print(f"ERROR: Failed to create environment: {e}")
        sys.exit(1)
    
    # Create trainer
    try:
        trainer = PPOTrainer(
            env=env,
            config=ppo_config,
            save_dir=save_dir,
            log_dir=log_dir
        )
        print("SUCCESS: PPO trainer created successfully")
    except Exception as e:
        print(f"ERROR: Failed to create trainer: {e}")
        sys.exit(1)
    
    # Resume from checkpoint if specified
    if args.resume:
        try:
            # TODO: Implement checkpoint loading in trainer
            print(f"INFO: Resume functionality not yet implemented")
            print(f"   Requested checkpoint: {args.resume}")
        except Exception as e:
            print(f"WARNING: Failed to resume from checkpoint: {e}")
            print("Starting fresh training...")
    
    # Print training configuration
    print("\nTraining Configuration:")
    print(f"   Algorithm: PPO (Proximal Policy Optimization)")
    print(f"   Environment: Pokemon Red (single instance with visuals)")
    print(f"   ROM: {args.rom}")
    print(f"   Max episode steps: {ppo_config.max_episode_steps:,}")
    print(f"   Batch horizon: {ppo_config.batch_horizon:,}")
    print(f"   PPO epochs: {ppo_config.ppo_epochs}")
    print(f"   Learning rates: policy={ppo_config.lr_policy:.0e}, value={ppo_config.lr_value:.0e}")
    print(f"   RND enabled: {ppo_config.use_rnd}")
    print(f"   Checkpoints enabled: {ppo_config.use_checkpoints}")
    print(f"   Curriculum stage: {ppo_config.curriculum_stage}")
    print(f"   Total epochs: {ppo_config.epochs:,}")
    print(f"   Save directory: {save_dir}")
    print(f"   Log directory: {log_dir}")
    print()
    
    if args.eval_only:
        print("Evaluation mode - not yet implemented")
        # TODO: Implement evaluation mode
        return
    
    # Start training
    try:
        print("Starting Pokemon Red speedrunning training!")
        print("   Goal: Learn to complete the game in minimum steps")
        print("   Progress will be tracked via badges, story events, and completion time")
        print("   Press Ctrl+C to stop training gracefully")
        print()
        
        trainer.train()
        
    except KeyboardInterrupt:
        print("\nSTOPPED:  Training stopped by user")
        print("Saving final checkpoint...")
        trainer._save_checkpoint("interrupted")
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("COMPLETED: Training completed successfully!")


if __name__ == "__main__":
    main()