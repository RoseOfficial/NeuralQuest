#!/usr/bin/env python3
"""Monitor NeuralQuest training progress in real-time."""

import time
import os
import csv
from pathlib import Path

def monitor_training():
    """Monitor training logs and checkpoints."""
    
    # Training artifacts
    logs_dir = Path("logs")
    checkpoints_dir = Path("checkpoints")
    metrics_file = logs_dir / "metrics.csv"
    
    print("NeuralQuest Training Monitor")
    print("=" * 50)
    
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
                                if len(values) >= 13:
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
            
            # Check for signs of learning
            if metrics_file.exists():
                print("Training is active - metrics being logged")
                break
            else:
                print("Training starting up...")
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            break
        except Exception as e:
            print(f"Monitor error: {e}")
        
        time.sleep(2)

if __name__ == "__main__":
    monitor_training()