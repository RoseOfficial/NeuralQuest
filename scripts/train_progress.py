#!/usr/bin/env python3
"""
Start progress-focused training for Pokemon Red.

This script demonstrates the enhanced progress-focused training mode
that uses a learned progress detector to identify meaningful state transitions
without hardcoding game-specific logic.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Start progress-focused training."""
    project_root = Path(__file__).parent.parent
    
    print("=" * 60)
    print("🎯 NeuralQuest Progress-Focused Training")
    print("=" * 60)
    print()
    print("This mode enhances exploration with learned progress detection:")
    print("• Learns to identify meaningful state transitions")
    print("• Combines intrinsic curiosity with progress rewards")
    print("• Uses progress-aware frontier sampling")
    print("• Maintains domain-agnostic principles")
    print()
    print("Starting training...")
    print()
    
    # Build command
    train_script = project_root / "scripts" / "train_pokemon.py"
    cmd = [
        sys.executable, str(train_script),
        "--mode", "progress",
        "--render",
        "--debug"
    ]
    
    # Execute training
    try:
        subprocess.run(cmd, check=True, cwd=project_root)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()