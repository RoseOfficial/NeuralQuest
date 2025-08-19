#!/usr/bin/env python3
"""Monitor NeuralQuest training progress in real-time with enhanced features."""

import time
import os
import csv
import subprocess
import webbrowser
from pathlib import Path
import argparse

def monitor_training(vector_mode=False, auto_open_viewer=False):
    """Monitor training logs and checkpoints."""
    
    # Training artifacts - handle vector training mode
    if vector_mode:
        logs_dir = Path("pokemon_vector_logs")
        checkpoints_dir = Path("pokemon_vector_checkpoints")
        metrics_file = logs_dir / "vector_metrics.csv"
        progress_screenshots = Path("progress_screenshots")
    else:
        logs_dir = Path("logs")
        checkpoints_dir = Path("checkpoints")
        metrics_file = logs_dir / "metrics.csv"
        progress_screenshots = None
    
    print("NeuralQuest Training Monitor")
    if vector_mode:
        print("Mode: Vector Training (Multi-Instance)")
    else:
        print("Mode: Single Instance Training")
    print("=" * 50)
    
    # Open progress viewer if requested and screenshots exist
    if auto_open_viewer and progress_screenshots and progress_screenshots.exists():
        viewer_path = Path("progress_viewer.html")
        if viewer_path.exists():
            print(f"Opening progress viewer: {viewer_path.absolute()}")
            webbrowser.open(f"file://{viewer_path.absolute()}")
        else:
            print("Progress viewer HTML not found, skipping auto-open")
    
    last_metrics = None
    last_checkpoint = None
    
    while True:
        try:
            # Check if training has started
            if not logs_dir.exists():
                print("Waiting for training to start...")
                time.sleep(5)
                continue
            
            # Monitor metrics
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:  # Header + at least one data row
                            latest_line = lines[-1].strip()
                            if latest_line != last_metrics:
                                # Parse latest metrics
                                values = latest_line.split(',')
                                if vector_mode and len(values) >= 15:
                                    # Vector training metrics
                                    epoch = values[0]
                                    steps = values[1]
                                    episodes = values[2]
                                    total_fps = float(values[3]) if values[3] else 0
                                    avg_fps = float(values[4]) if values[4] else 0
                                    reward = float(values[5]) if values[5] else 0
                                    length = float(values[6]) if values[6] else 0
                                    archive_size = values[13]
                                    n_envs = values[18] if len(values) > 18 else "10"
                                    
                                    print(f"Epoch {epoch:>6} | Steps {steps:>8} | Episodes {episodes:>6} | "
                                          f"Total FPS {total_fps:>6.1f} | Avg FPS {avg_fps:>5.1f} | "
                                          f"Reward {reward:>7.3f} | Length {length:>5.1f} | Archive {archive_size:>6} | Envs {n_envs}")
                                    
                                elif not vector_mode and len(values) >= 13:
                                    # Single training metrics
                                    epoch = values[0]
                                    steps = values[1]
                                    episodes = values[2]
                                    fps = float(values[3]) if values[3] else 0
                                    archive_size = values[12]
                                    
                                    print(f"Epoch {epoch:>6} | Steps {steps:>8} | Episodes {episodes:>6} | FPS {fps:>6.1f} | Archive {archive_size:>6}")
                                    
                                last_metrics = latest_line
                except Exception as e:
                    print(f"Error reading metrics: {e}")
            
            # Monitor checkpoints
            if checkpoints_dir.exists():
                checkpoints = list(checkpoints_dir.glob("epoch_*"))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                    if latest_checkpoint != last_checkpoint:
                        print(f"New checkpoint: {latest_checkpoint.name}")
                        last_checkpoint = latest_checkpoint
            
            
            # Monitor progress screenshots (vector mode only)
            if progress_screenshots and progress_screenshots.exists():
                screenshot_count = len(list(progress_screenshots.glob("*/latest.png")))
                if screenshot_count > 0:
                    print(f"Progress screenshots: {screenshot_count} environments captured")
            
            # Check for signs of learning
            if metrics_file.exists():
                print("Training is active - metrics being logged")
                if vector_mode and progress_screenshots and progress_screenshots.exists():
                    print(f"📸 Screenshots available in: {progress_screenshots}")
                    print("🌐 Open progress_viewer.html to see real-time progress")
                break
            else:
                print("Training starting up...")
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            break
        except Exception as e:
            print(f"Monitor error: {e}")
        
        time.sleep(2)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Monitor NeuralQuest training progress")
    parser.add_argument(
        "--vector",
        action="store_true",
        help="Monitor vector training (multi-instance mode)"
    )
    parser.add_argument(
        "--open-viewer",
        action="store_true",
        help="Auto-open progress viewer in browser (vector mode only)"
    )
    
    args = parser.parse_args()
    
    monitor_training(
        vector_mode=args.vector,
        auto_open_viewer=args.open_viewer
    )


if __name__ == "__main__":
    main()