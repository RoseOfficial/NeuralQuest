#!/usr/bin/env python3
"""
Pokemon Red Evaluation Script for NeuralQuest

Convenience script for evaluating trained agents on Pokemon Red
with analysis of exploration patterns and gameplay progression.
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from redai.cli.eval import main as eval_main


def get_pokemon_rom_path():
    """Get the path to Pokemon Red ROM."""
    project_dir = Path(__file__).parent.parent
    rom_path = project_dir / "roms" / "pokemon_red.gb"
    
    if not rom_path.exists():
        print(f"Error: Pokemon Red ROM not found at {rom_path}")
        print("Please ensure 'pokemon_red.gb' is in the roms/ directory.")
        sys.exit(1)
    
    return str(rom_path)


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in a directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for epoch directories
    epoch_dirs = []
    for item in os.listdir(checkpoint_dir):
        item_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(item_path) and item.startswith("epoch_"):
            try:
                epoch_num = int(item.split("_")[1])
                epoch_dirs.append((epoch_num, item_path))
            except (IndexError, ValueError):
                continue
    
    if not epoch_dirs:
        return None
    
    # Return the latest epoch
    epoch_dirs.sort()
    return epoch_dirs[-1][1]


def main():
    """Main evaluation script for Pokemon Red."""
    parser = argparse.ArgumentParser(
        description="Evaluate NeuralQuest agent on Pokemon Red",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (auto-finds latest if not specified)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["standard", "fast", "exploration"],
        default="standard",
        help="Which training mode checkpoints to look for"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes to evaluate"
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering (show gameplay)"
    )
    
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy (no exploration)"
    )
    
    parser.add_argument(
        "--analyze-archive",
        action="store_true",
        help="Analyze archive statistics and exploration patterns"
    )
    
    parser.add_argument(
        "--frontier-analysis",
        action="store_true", 
        help="Detailed frontier cell analysis"
    )
    
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Save evaluation results to JSON file"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Get ROM path
    rom_path = get_pokemon_rom_path()
    
    # Find checkpoint if not specified
    if args.checkpoint is None:
        # Look for checkpoints based on mode
        checkpoint_dirs = {
            "standard": "pokemon_checkpoints",
            "fast": "pokemon_fast_checkpoints", 
            "exploration": "pokemon_exploration_checkpoints"
        }
        
        checkpoint_dir = checkpoint_dirs[args.mode]
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        
        if latest_checkpoint is None:
            print(f"❌ No checkpoints found in {checkpoint_dir}")
            print(f"💡 Train first with: python train_pokemon.py --mode {args.mode}")
            sys.exit(1)
        
        args.checkpoint = latest_checkpoint
        print(f"📁 Using latest checkpoint: {args.checkpoint}")
    
    print(f"🎮 Evaluating NeuralQuest on Pokemon Red")
    print(f"🤖 Checkpoint: {args.checkpoint}")
    print(f"🎯 Episodes: {args.episodes}")
    
    if args.render:
        print("👁️  Rendering enabled")
    
    if args.deterministic:
        print("🎯 Deterministic policy")
    
    # Build command line arguments for evaluation script
    eval_args = [rom_path, "--checkpoint", args.checkpoint]
    
    eval_args.extend(["--episodes", str(args.episodes)])
    
    if args.render:
        eval_args.append("--render")
    
    if args.deterministic:
        eval_args.append("--deterministic")
    
    if args.analyze_archive:
        eval_args.append("--analyze-archive")
    
    if args.frontier_analysis:
        eval_args.append("--frontier-analysis")
    
    if args.save_results:
        eval_args.extend(["--save-stats", args.save_results])
    elif args.analyze_archive or args.frontier_analysis:
        # Auto-generate results file if doing analysis
        results_file = f"pokemon_eval_results_{args.mode}.json"
        eval_args.extend(["--save-stats", results_file])
        print(f"💾 Results will be saved to: {results_file}")
    
    if args.debug:
        eval_args.append("--debug")
    
    # Set up sys.argv for the evaluation script
    original_argv = sys.argv
    sys.argv = ["eval.py"] + eval_args
    
    try:
        # Run the evaluation
        eval_main()
        
        print("\n✅ Evaluation completed successfully!")
        
        # Print additional Pokemon-specific insights
        if args.analyze_archive or args.frontier_analysis:
            print("\n🔍 Pokemon Red Exploration Analysis:")
            print("   - Check archive size for world coverage")
            print("   - High visit count areas may indicate interesting gameplay")
            print("   - Frontier cells show unexplored potential")
            print("   - Episode length indicates survival/progression ability")
            
    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


if __name__ == "__main__":
    print("🧠 NeuralQuest - Pokemon Red Evaluation")
    print("=" * 50)
    main()