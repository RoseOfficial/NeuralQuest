#!/usr/bin/env python3
"""
Vector training script for NeuralQuest with multiple PyBoy instances.

Runs 10 PyBoy instances (9 headless + 1 with visuals) all connected to the
same neural network for efficient parallel training.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from redai.train.vector_trainer import VectorTrainer
from redai.train.config import Config


def get_pokemon_rom_path():
    """Get the path to Pokemon Red ROM."""
    project_dir = Path(__file__).parent
    rom_path = project_dir / "roms" / "pokemon_red.gb"
    
    if not rom_path.exists():
        print(f"Error: Pokemon Red ROM not found at {rom_path}")
        print("Please ensure 'pokemon_red.gb' is in the roms/ directory.")
        print("You must legally own the ROM file.")
        sys.exit(1)
    
    return str(rom_path)


def main():
    """Main vector training script."""
    parser = argparse.ArgumentParser(
        description="Train NeuralQuest agent with multiple PyBoy instances",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pokemon_red_vector_optimal.toml",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--n-envs",
        type=int,
        default=10,
        help="Number of parallel PyBoy instances"
    )
    
    parser.add_argument(
        "--visual-env",
        type=int,
        default=0,
        help="Which environment shows visuals (others headless, -1 for all headless)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint directory"
    )
    
    parser.add_argument(
        "--monitor-progress",
        action="store_true",
        help="Enable progress monitoring with periodic screenshots"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs"
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
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--track-events",
        action="store_true",
        help="Enable Pokemon event tracking (naming, catching, badges)"
    )
    
    parser.add_argument(
        "--event-log-dir",
        type=str,
        default="pokemon_events",
        help="Directory to save Pokemon event logs"
    )
    
    args = parser.parse_args()
    
    # Get ROM path
    rom_path = get_pokemon_rom_path()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = Config.from_toml(args.config)
    
    # Apply command line overrides
    if args.epochs:
        config.train.epochs = args.epochs
    
    config.train.n_envs = args.n_envs
    config.train.visual_env_idx = args.visual_env
    
    # Apply additional overrides
    overrides = {}
    for override_str in args.override:
        if "=" not in override_str:
            print(f"Invalid override format: {override_str}")
            continue
        key, value = override_str.split("=", 1)
        # Try to parse as number
        try:
            if "." in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            # Keep as string if not a number
            if value.lower() in ["true", "false"]:
                value = value.lower() == "true"
    
        overrides[key] = value
    
    if overrides:
        config.override(overrides)
        print(f"Applied overrides: {overrides}")
    
    # Setup debug logging
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    print("NeuralQuest Vector Training")
    print("=" * 50)
    print(f"ROM: {rom_path}")
    print(f"Config: {args.config}")
    print(f"Environments: {args.n_envs}")
    print(f"Visual environment: #{args.visual_env}")
    print(f"Headless environments: {args.n_envs - 1}")
    
    if args.resume:
        print(f"Resuming from: {args.resume}")
    
    # Initialize trainer
    trainer = VectorTrainer(
        config=config,
        n_envs=args.n_envs,
        visual_env_idx=args.visual_env,
        monitor_progress=args.monitor_progress,
        track_events=args.track_events,
        event_log_dir=args.event_log_dir
    )
    
    try:
        # Start training
        print("\nStarting vector training...")
        trainer.train(rom_path=rom_path, resume_path=args.resume)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up
        trainer.close()
    
    print("Vector training completed!")


if __name__ == "__main__":
    # Windows multiprocessing compatibility (disabled - causes hang with PyBoy)
    # if sys.platform.startswith('win'):
    #     import multiprocessing
    #     multiprocessing.set_start_method('spawn', force=True)
    
    main()