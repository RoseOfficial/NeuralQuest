#!/usr/bin/env python3
"""
Create a savestate that starts in the actual game (past intro sequence).
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from redai.envs.pyboy_env import Env

def main():
    """Create a proper starting savestate."""
    
    rom_path = project_root / "roms" / "pokemon_red.gb"
    
    print("Creating environment to navigate past intro...")
    print("This will open the game window - watch the progress")
    
    env = Env(
        rom_path=str(rom_path),
        frame_skip=4,
        sticky_p=0.0,
        max_episode_steps=10000,  # Longer to get through intro
        deterministic=True,
        headless=False,  # Show progress
        seed=1337
    )
    
    obs = env.reset()
    print("Game started - beginning intro sequence navigation...")
    
    # Navigate through Pokemon Red intro sequence
    # This is a rough sequence to get past the intro
    
    actions_sequence = [
        # Skip through Professor Oak's speech
        ("A", 30),      # Start game / skip text
        ("A", 30),      # Continue through Oak's speech
        ("A", 30),      # More speech
        ("A", 30),      # More speech
        ("A", 30),      # "This is a Pokemon!"
        ("A", 30),      # Continue
        ("A", 30),      # Name rival
        ("A", 30),      # Default name
        ("A", 30),      # Your name?
        ("A", 30),      # Default name
        ("A", 30),      # Continue
        ("A", 50),      # Final intro bits
        ("start", 5),   # Sometimes need start
        ("A", 100),     # Should get us to the actual game
    ]
    
    print("Executing intro sequence...")
    for action_name, repeats in actions_sequence:
        if action_name in env.ACTIONS:
            action_idx = env.ACTIONS.index(action_name)
            print(f"Pressing {action_name} {repeats} times...")
            
            for _ in range(repeats):
                obs, reward, done, info = env.step(action_idx)
                if done:
                    print("Episode ended during intro - this is expected")
                    break
            time.sleep(0.1)  # Small pause to see progress
    
    print("\nTaking random actions to ensure we're in the overworld...")
    # Take some random actions to make sure we're actually playing
    for i in range(200):
        action = np.random.randint(0, len(env.ACTIONS))
        obs, reward, done, info = env.step(action)
        if done:
            break
        
        if i % 50 == 0:
            print(f"Random action {i}/200...")
    
    # Check if we can read player position now
    try:
        x_pos = env._pyboy.memory[0xD362]
        y_pos = env._pyboy.memory[0xD361]
        map_id = env._pyboy.memory[0xD35E]
        print(f"Final position: X={x_pos}, Y={y_pos}, Map={map_id}")
        
        if x_pos != 0 or y_pos != 0 or map_id != 0:
            print("SUCCESS: Player position changed - we're in the actual game!")
        else:
            print("WARNING: Position still 0,0,0 - might still be in intro")
    except:
        print("Could not read position - continuing anyway")
    
    # Save the state
    print("\nSaving starting game state...")
    savestate = env.save_state()
    
    # Save to file
    savestate_path = project_root / "game_start.state"
    with open(savestate_path, "wb") as f:
        f.write(savestate)
    
    print(f"Saved game start state to: {savestate_path}")
    print(f"State size: {len(savestate)} bytes")
    print("\nThis savestate can now be used as the starting point for training")
    print("to ensure the agent starts in the actual game world, not the intro.")
    
    time.sleep(2)
    env.close()

if __name__ == "__main__":
    main()