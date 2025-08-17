"""Command-line interface for training NeuralQuest agent."""

import argparse
import os
import sys
from typing import Dict, Any

from ..train.trainer import Trainer
from ..train.config import Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train NeuralQuest agent on Game Boy games",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "rom_path",
        type=str,
        help="Path to Game Boy ROM file (.gb or .gbc)"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML configuration file"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from"
    )
    
    # Override parameters
    parser.add_argument(
        "--override",
        type=str,
        action="append",
        default=[],
        help="Override config parameters (e.g., --override rnd.beta=0.3)"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    # Output directories
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for logs and metrics"
    )
    
    # Evaluation
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation, no training"
    )
    
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation"
    )
    
    # Debugging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose output"
    )
    
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run short smoke test (5 minutes)"
    )
    
    return parser.parse_args()


def parse_overrides(override_strings: list) -> Dict[str, Any]:
    """
    Parse configuration override strings.
    
    Args:
        override_strings: List of override strings like "section.param=value"
        
    Returns:
        Dictionary of parsed overrides
    """
    overrides = {}
    
    for override_str in override_strings:
        if "=" not in override_str:
            raise ValueError(f"Invalid override format: {override_str}. Expected 'section.param=value'")
        
        key, value_str = override_str.split("=", 1)
        
        # Try to parse value as different types
        if value_str.lower() in ("true", "false"):
            value = value_str.lower() == "true"
        elif value_str.isdigit():
            value = int(value_str)
        else:
            try:
                value = float(value_str)
            except ValueError:
                value = value_str  # Keep as string
        
        overrides[key] = value
    
    return overrides


def setup_config(args) -> Config:
    """
    Setup configuration from args and config file.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configured Config object
    """
    # Load base configuration
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        config = Config.from_toml(args.config)
        print(f"Loaded configuration from {args.config}")
    else:
        config = Config.default()
        print("Using default configuration")
    
    # Apply command line overrides
    if args.epochs is not None:
        config.train.epochs = args.epochs
    
    if args.seed is not None:
        config.env.seed = args.seed
    
    if args.save_dir:
        config.train.save_dir = args.save_dir
    
    if args.log_dir:
        config.train.log_dir = args.log_dir
    
    # Smoke test adjustments
    if args.smoke_test:
        config.train.epochs = 10
        config.algo.batch_horizon = 128
        config.train.ckpt_every = 5
        config.train.log_every = 1
        print("Smoke test mode: reduced epochs and batch size")
    
    # Apply parameter overrides
    if args.override:
        overrides = parse_overrides(args.override)
        config.override(overrides)
        print(f"Applied overrides: {overrides}")
    
    return config


def validate_rom_path(rom_path: str) -> None:
    """Validate ROM file path."""
    if not os.path.exists(rom_path):
        raise FileNotFoundError(f"ROM file not found: {rom_path}")
    
    if not rom_path.lower().endswith(('.gb', '.gbc')):
        print(f"Warning: ROM file does not have .gb or .gbc extension: {rom_path}")


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Validate ROM path
        validate_rom_path(args.rom_path)
        
        # Setup configuration
        config = setup_config(args)
        
        # Create trainer
        trainer = Trainer(config)
        
        # Resume from checkpoint if specified
        if args.resume:
            if not os.path.exists(args.resume):
                raise FileNotFoundError(f"Checkpoint directory not found: {args.resume}")
            
            # Setup environment first (required for loading networks)
            trainer.setup_environment(args.rom_path)
            trainer.load_checkpoint(args.resume)
            print(f"Resumed from checkpoint: {args.resume}")
        
        # Run evaluation only
        if args.eval_only:
            if not args.resume:
                raise ValueError("--eval-only requires --resume checkpoint")
            
            print(f"Running evaluation for {args.eval_episodes} episodes...")
            eval_stats = trainer.evaluate(args.eval_episodes)
            
            print("\nEvaluation Results:")
            for key, value in eval_stats.items():
                print(f"  {key}: {value:.4f}")
            
            return
        
        # Print configuration
        if args.debug:
            print("\nConfiguration:")
            print(f"  Environment: frame_skip={config.env.frame_skip}, "
                  f"sticky_p={config.env.sticky_p}, seed={config.env.seed}")
            print(f"  Algorithm: lr_policy={config.algo.lr_policy}, "
                  f"lr_value={config.algo.lr_value}, gamma={config.algo.gamma}")
            print(f"  RND: beta={config.rnd.beta}, lr={config.rnd.lr}")
            print(f"  Archive: capacity={config.archive.capacity}, "
                  f"p_frontier={config.archive.p_frontier}")
            print(f"  Training: epochs={config.train.epochs}, "
                  f"batch_horizon={config.algo.batch_horizon}")
            print()
        
        # Start training
        print(f"Starting training with ROM: {args.rom_path}")
        print(f"Checkpoints will be saved to: {config.train.save_dir}")
        print(f"Logs will be saved to: {config.train.log_dir}")
        print()
        
        trainer.train(args.rom_path)
        
        print("\nTraining completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()