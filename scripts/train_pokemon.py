#!/usr/bin/env python3
"""
Pokemon Red Training Script for NeuralQuest

Convenience script for training the domain-agnostic RL agent on Pokemon Red
with various preset configurations optimized for different research objectives.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from redai.cli.run_train import main as train_main


def get_pokemon_rom_path():
    """Get the path to Pokemon Red ROM."""
    project_dir = Path(__file__).parent.parent
    rom_path = project_dir / "roms" / "pokemon_red.gb"
    
    if not rom_path.exists():
        print(f"Error: Pokemon Red ROM not found at {rom_path}")
        print("Please ensure 'pokemon_red.gb' is in the roms/ directory.")
        print("You must legally own the ROM file.")
        sys.exit(1)
    
    return str(rom_path)


def main():
    """Main training script for Pokemon Red."""
    parser = argparse.ArgumentParser(
        description="Train NeuralQuest agent on Pokemon Red",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        choices=["standard", "fast", "exploration", "smoke"],
        default="standard",
        help="Training mode with preset configurations"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint directory"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs"
    )
    
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation (requires --resume)"
    )
    
    parser.add_argument(
        "--override",
        type=str,
        action="append",
        default=[],
        help="Override config parameters (e.g., --override rnd.beta=0.3)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Get ROM path
    rom_path = get_pokemon_rom_path()
    
    # Select configuration based on mode
    config_map = {
        "standard": "configs/pokemon_red.toml",
        "fast": "configs/pokemon_red_fast.toml", 
        "exploration": "configs/pokemon_red_exploration.toml",
        "smoke": "configs/smoke_test.toml"
    }
    
    config_path = config_map[args.mode]
    
    print(f"Training NeuralQuest on Pokemon Red")
    print(f"ROM: {rom_path}")
    print(f"Mode: {args.mode}")
    print(f"Config: {config_path}")
    
    if args.resume:
        print(f"Resuming from: {args.resume}")
    
    # Build command line arguments for the main training script
    train_args = [rom_path, "--config", config_path]
    
    if args.resume:
        train_args.extend(["--resume", args.resume])
    
    if args.epochs:
        train_args.extend(["--epochs", str(args.epochs)])
    
    if args.eval_only:
        train_args.append("--eval-only")
    
    if args.debug:
        train_args.append("--debug")
    
    if args.mode == "smoke":
        train_args.append("--smoke-test")
    
    for override in args.override:
        train_args.extend(["--override", override])
    
    # Set up sys.argv for the training script
    original_argv = sys.argv
    sys.argv = ["run_train.py"] + train_args
    
    try:
        # Run the training
        train_main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


if __name__ == "__main__":
    print("NeuralQuest - Pokemon Red Training")
    print("=" * 50)
    main()