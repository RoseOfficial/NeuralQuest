"""Command-line interface for evaluating trained NeuralQuest agents."""

import argparse
import os
import sys
import time
from typing import Dict, Any

from ..train.trainer import Trainer
from ..train.config import Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained NeuralQuest agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "rom_path",
        type=str,
        help="Path to Game Boy ROM file (.gb or .gbc)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of episodes to evaluate"
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render gameplay (non-headless mode)"
    )
    
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Path to save gameplay video"
    )
    
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy (no exploration)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for evaluation"
    )
    
    # Analysis options
    parser.add_argument(
        "--analyze-archive",
        action="store_true",
        help="Analyze archive statistics"
    )
    
    parser.add_argument(
        "--frontier-analysis",
        action="store_true",
        help="Analyze frontier cells"
    )
    
    parser.add_argument(
        "--save-stats",
        type=str,
        default=None,
        help="Path to save evaluation statistics (JSON)"
    )
    
    # Debugging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose output"
    )
    
    return parser.parse_args()


def analyze_archive(trainer: Trainer) -> Dict[str, Any]:
    """
    Analyze archive statistics and frontier cells.
    
    Args:
        trainer: Trainer instance with loaded checkpoint
        
    Returns:
        Archive analysis results
    """
    if trainer.archive is None:
        return {"error": "No archive found in checkpoint"}
    
    # Basic archive statistics
    stats = trainer.archive.stats()
    
    # Frontier analysis
    frontier_cells = trainer.archive.get_frontier_cells(num_cells=20)
    frontier_analysis = {
        "num_frontier_cells": len(frontier_cells),
        "frontier_visit_counts": [cell.visit_count for cell in frontier_cells],
        "frontier_ages": [trainer.archive.total_steps - cell.first_seen_step for cell in frontier_cells],
        "frontier_scores": [trainer.archive._compute_frontier_score(cell, trainer.archive.total_steps) 
                           for cell in frontier_cells]
    }
    
    # Cell distribution analysis
    all_cells = list(trainer.archive.cells.values())
    if all_cells:
        visit_distribution = {
            "min_visits": min(cell.visit_count for cell in all_cells),
            "max_visits": max(cell.visit_count for cell in all_cells),
            "mean_visits": sum(cell.visit_count for cell in all_cells) / len(all_cells),
            "total_visits": sum(cell.visit_count for cell in all_cells)
        }
        
        age_distribution = {
            "oldest_cell": max(trainer.archive.total_steps - cell.first_seen_step for cell in all_cells),
            "newest_cell": min(trainer.archive.total_steps - cell.first_seen_step for cell in all_cells),
            "mean_age": sum(trainer.archive.total_steps - cell.first_seen_step for cell in all_cells) / len(all_cells)
        }
    else:
        visit_distribution = age_distribution = {}
    
    return {
        "basic_stats": stats,
        "frontier_analysis": frontier_analysis,
        "visit_distribution": visit_distribution,
        "age_distribution": age_distribution
    }


def run_evaluation(trainer: Trainer, rom_path: str, args) -> Dict[str, Any]:
    """
    Run policy evaluation.
    
    Args:
        trainer: Trainer instance
        rom_path: Path to ROM file
        args: Command line arguments
        
    Returns:
        Evaluation results
    """
    print(f"Running evaluation for {args.episodes} episodes...")
    
    # Setup environment for evaluation
    if not hasattr(trainer, 'env') or trainer.env is None:
        trainer.setup_environment(rom_path)
    
    # Configure environment for evaluation
    if args.render:
        # Create new environment with rendering enabled
        from ..envs.pyboy_env import Env
        eval_env = Env(
            rom_path=rom_path,
            frame_skip=trainer.config.env.frame_skip,
            sticky_p=0.0 if args.deterministic else trainer.config.env.sticky_p,
            max_episode_steps=trainer.config.env.max_episode_steps,
            deterministic=True,
            headless=False,  # Enable rendering
            seed=args.seed
        )
    else:
        eval_env = trainer.env
        eval_env.seed(args.seed)
    
    episode_rewards = []
    episode_lengths = []
    intrinsic_rewards = []
    
    start_time = time.time()
    
    for episode in range(args.episodes):
        obs = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        episode_intrinsic = 0
        done = False
        
        print(f"Episode {episode + 1}/{args.episodes}", end=" ", flush=True)
        
        while not done:
            # Get action from policy
            action, log_prob, value, aux_info = trainer.actor_critic.act(obs)
            
            # For deterministic evaluation, use greedy action selection
            if args.deterministic:
                action_probs = aux_info['action_probs']
                action = int(np.argmax(action_probs))
            
            # Step environment
            next_obs, env_reward, done, info = eval_env.step(action)
            
            # Compute intrinsic reward for analysis
            if trainer.rnd is not None:
                intrinsic_reward = trainer.rnd.intrinsic_reward(next_obs)
                episode_intrinsic += intrinsic_reward
            
            # Use environment reward (should be 0 for NeuralQuest)
            episode_reward += env_reward
            episode_length += 1
            
            obs = next_obs
            
            if episode_length >= trainer.config.env.max_episode_steps:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        intrinsic_rewards.append(episode_intrinsic)
        
        print(f"Reward: {episode_reward:.2f}, Length: {episode_length}, "
              f"Intrinsic: {episode_intrinsic:.2f}")
    
    eval_time = time.time() - start_time
    
    # Close evaluation environment if different from training env
    if args.render and eval_env != trainer.env:
        eval_env.close()
    
    # Compute statistics
    import numpy as np
    
    results = {
        "num_episodes": args.episodes,
        "eval_time_seconds": eval_time,
        "episode_rewards": {
            "mean": float(np.mean(episode_rewards)),
            "std": float(np.std(episode_rewards)),
            "min": float(np.min(episode_rewards)),
            "max": float(np.max(episode_rewards)),
            "median": float(np.median(episode_rewards))
        },
        "episode_lengths": {
            "mean": float(np.mean(episode_lengths)),
            "std": float(np.std(episode_lengths)),
            "min": float(np.min(episode_lengths)),
            "max": float(np.max(episode_lengths)),
            "median": float(np.median(episode_lengths))
        },
        "intrinsic_rewards": {
            "mean": float(np.mean(intrinsic_rewards)),
            "std": float(np.std(intrinsic_rewards)),
            "min": float(np.min(intrinsic_rewards)),
            "max": float(np.max(intrinsic_rewards)),
            "median": float(np.median(intrinsic_rewards))
        }
    }
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Validate inputs
        if not os.path.exists(args.rom_path):
            raise FileNotFoundError(f"ROM file not found: {args.rom_path}")
        
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint}")
        
        # Load checkpoint
        print(f"Loading checkpoint from {args.checkpoint}")
        
        # Create trainer with default config (will be overridden by checkpoint)
        config = Config.default()
        trainer = Trainer(config)
        
        # Setup environment and load checkpoint
        trainer.setup_environment(args.rom_path)
        trainer.load_checkpoint(args.checkpoint)
        
        print("Checkpoint loaded successfully")
        
        # Run evaluation
        eval_results = run_evaluation(trainer, args.rom_path, args)
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        print(f"Episodes: {eval_results['num_episodes']}")
        print(f"Evaluation time: {eval_results['eval_time_seconds']:.1f} seconds")
        print()
        
        print("Episode Rewards:")
        for key, value in eval_results['episode_rewards'].items():
            print(f"  {key}: {value:.4f}")
        print()
        
        print("Episode Lengths:")
        for key, value in eval_results['episode_lengths'].items():
            print(f"  {key}: {value:.2f}")
        print()
        
        print("Intrinsic Rewards:")
        for key, value in eval_results['intrinsic_rewards'].items():
            print(f"  {key}: {value:.4f}")
        print()
        
        # Archive analysis
        if args.analyze_archive or args.frontier_analysis:
            print("="*60)
            print("ARCHIVE ANALYSIS")
            print("="*60)
            
            archive_analysis = analyze_archive(trainer)
            
            if "error" in archive_analysis:
                print(f"Error: {archive_analysis['error']}")
            else:
                basic_stats = archive_analysis['basic_stats']
                print(f"Archive size: {basic_stats['size']}")
                print(f"Cells added: {basic_stats['cells_added']}")
                print(f"Cells evicted: {basic_stats['cells_evicted']}")
                print(f"Capacity used: {basic_stats['capacity_used']:.2%}")
                print(f"Average visit count: {basic_stats['avg_visit_count']:.2f}")
                
                if args.frontier_analysis:
                    frontier = archive_analysis['frontier_analysis']
                    print(f"\nFrontier cells: {frontier['num_frontier_cells']}")
                    if frontier['frontier_visit_counts']:
                        print(f"Frontier visit counts: min={min(frontier['frontier_visit_counts'])}, "
                              f"max={max(frontier['frontier_visit_counts'])}, "
                              f"mean={sum(frontier['frontier_visit_counts'])/len(frontier['frontier_visit_counts']):.1f}")
        
        # Save results
        if args.save_stats:
            import json
            
            full_results = {
                "evaluation": eval_results,
                "archive_analysis": analyze_archive(trainer) if args.analyze_archive else None,
                "checkpoint": args.checkpoint,
                "rom_path": args.rom_path,
                "evaluation_args": vars(args)
            }
            
            with open(args.save_stats, 'w') as f:
                json.dump(full_results, f, indent=2)
            
            print(f"\nResults saved to {args.save_stats}")
        
        print("\nEvaluation completed successfully!")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()