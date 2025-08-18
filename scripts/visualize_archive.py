#!/usr/bin/env python3
"""Visualize archive states to see what game locations are being discovered."""

import sys
import os
from pathlib import Path
import numpy as np
import argparse
from PIL import Image
import io

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from redai.envs.pyboy_env import Env


def visualize_archive_states(archive_path: str, output_dir: str = "archive_screenshots", 
                           max_states: int = 50, sort_by: str = "frontier"):
    """
    Visualize states from the archive to see what game locations are being discovered.
    
    Args:
        archive_path: Path to the archive .npz file or checkpoint directory
        output_dir: Directory to save screenshot images
        max_states: Maximum number of states to visualize
        sort_by: How to sort states ('frontier', 'visits', 'discovery_time', 'recent')
    """
    print(f"Visualizing archive states from: {archive_path}")
    
    # Load archive data
    if os.path.isdir(archive_path):
        # It's a checkpoint directory
        archive_file = os.path.join(archive_path, "archive.npz")
    else:
        # It's a direct path to archive file
        archive_file = archive_path
    
    if not os.path.exists(archive_file):
        print(f"Archive file not found: {archive_file}")
        return
    
    try:
        data = np.load(archive_file, allow_pickle=True)
        print(f"Loaded archive with keys: {list(data.keys())}")
    except Exception as e:
        print(f"Error loading archive: {e}")
        return
    
    # Extract cells data
    if 'cells' not in data:
        print("No 'cells' data found in archive")
        return
    
    cells = data['cells'].item()
    current_step = int(data.get('total_steps', 0))
    
    print(f"Found {len(cells)} cells in archive")
    print(f"Current training step: {current_step}")
    
    # Sort cells based on criteria
    sorted_cells = sort_cells(cells, current_step, sort_by)
    
    # Limit to max_states
    cells_to_visualize = sorted_cells[:max_states]
    print(f"Visualizing top {len(cells_to_visualize)} states sorted by {sort_by}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get ROM path for PyBoy initialization
    rom_path = find_pokemon_rom()
    if not rom_path:
        print("Pokemon ROM not found - cannot visualize states")
        return
    
    # Initialize PyBoy for state visualization
    print("Initializing PyBoy for state visualization...")
    env = Env(
        rom_path=rom_path,
        headless=True,  # Always headless for visualization
        deterministic=True
    )
    
    # Visualize each state
    print("Rendering states...")
    success_count = 0
    
    for i, (cell_id, cell_data) in enumerate(cells_to_visualize):
        try:
            # Load the state
            savestate = cell_data['savestate']
            env.load_state(savestate)
            
            # Get screenshot
            screen_rgb = env.render_rgb()
            
            # Create filename with metadata
            visit_count = cell_data['visit_count']
            first_step = cell_data['first_seen_step']
            last_step = cell_data['last_seen_step']
            
            filename = f"state_{i:03d}_id{cell_id}_v{visit_count}_s{first_step}-{last_step}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Save screenshot
            image = Image.fromarray(screen_rgb)
            image.save(filepath)
            
            success_count += 1
            
            if i % 10 == 0:
                print(f"  Rendered {i+1}/{len(cells_to_visualize)} states...")
                
        except Exception as e:
            print(f"  Error rendering state {i} (ID: {cell_id}): {e}")
            continue
    
    print(f"\nSuccessfully rendered {success_count}/{len(cells_to_visualize)} states")
    print(f"Screenshots saved to: {output_dir}")
    
    # Generate summary HTML
    generate_summary_html(cells_to_visualize, output_dir, sort_by, current_step)
    
    # Clean up
    env.close()


def sort_cells(cells, current_step, sort_by):
    """Sort cells by the specified criteria."""
    cell_list = list(cells.items())
    
    if sort_by == "frontier":
        # Sort by frontier score (like the archive does)
        def frontier_score(cell_data):
            visit_factor = 1.0 / (1.0 + cell_data['visit_count'])
            progression_factor = min(1.0, cell_data['first_seen_step'] / max(1000, current_step * 0.1))
            age = current_step - cell_data['first_seen_step']
            if age < 50:
                recency_factor = 0.3
            elif age < 200:
                recency_factor = 0.8
            else:
                recency_factor = 1.0
            
            return 0.5 * visit_factor + 0.4 * progression_factor + 0.1 * recency_factor
        
        cell_list.sort(key=lambda x: frontier_score(x[1]), reverse=True)
        
    elif sort_by == "visits":
        # Sort by visit count (ascending - less visited first)
        cell_list.sort(key=lambda x: x[1]['visit_count'])
        
    elif sort_by == "discovery_time":
        # Sort by when discovered (latest first)
        cell_list.sort(key=lambda x: x[1]['first_seen_step'], reverse=True)
        
    elif sort_by == "recent":
        # Sort by when last visited (most recent first)
        cell_list.sort(key=lambda x: x[1]['last_seen_step'], reverse=True)
    
    return cell_list


def generate_summary_html(cells_to_visualize, output_dir, sort_by, current_step):
    """Generate an HTML summary page for the visualized states."""
    html_path = os.path.join(output_dir, "summary.html")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Archive States Visualization - Sorted by {sort_by}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #1a1a1a; color: #e0e0e0; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .stats {{ margin-bottom: 20px; padding: 15px; background-color: #2d2d2d; border-radius: 5px; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }}
            .state-item {{ 
                background-color: #2d2d2d; border: 1px solid #444; border-radius: 5px; 
                padding: 10px; text-align: center; 
            }}
            .state-item img {{ 
                width: 160px; height: 144px; 
                image-rendering: pixelated; border: 1px solid #555; 
            }}
            .state-info {{ font-size: 12px; margin-top: 5px; color: #ccc; }}
            .state-title {{ font-weight: bold; color: #4CAF50; margin-bottom: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎮 Archive States Visualization</h1>
            <p>Discovered game locations sorted by {sort_by}</p>
        </div>
        
        <div class="stats">
            <strong>Training Step:</strong> {current_step:,}<br>
            <strong>States Visualized:</strong> {len(cells_to_visualize)}<br>
            <strong>Sort Method:</strong> {sort_by}
        </div>
        
        <div class="grid">
    """
    
    for i, (cell_id, cell_data) in enumerate(cells_to_visualize):
        visit_count = cell_data['visit_count']
        first_step = cell_data['first_seen_step']
        last_step = cell_data['last_seen_step']
        
        filename = f"state_{i:03d}_id{cell_id}_v{visit_count}_s{first_step}-{last_step}.png"
        
        html_content += f"""
            <div class="state-item">
                <div class="state-title">State #{i+1}</div>
                <img src="{filename}" alt="State {i+1}">
                <div class="state-info">
                    ID: {cell_id}<br>
                    Visits: {visit_count}<br>
                    Steps: {first_step:,} - {last_step:,}
                </div>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Summary HTML created: {html_path}")


def find_pokemon_rom():
    """Find Pokemon Red ROM file."""
    possible_paths = [
        "roms/pokemon_red.gb",
        "../roms/pokemon_red.gb",
        "../../roms/pokemon_red.gb",
        "pokemon_red.gb"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Visualize archive states")
    parser.add_argument(
        "archive_path",
        help="Path to archive .npz file or checkpoint directory"
    )
    parser.add_argument(
        "--output-dir",
        default="archive_screenshots",
        help="Directory to save screenshots"
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=50,
        help="Maximum number of states to visualize"
    )
    parser.add_argument(
        "--sort-by",
        choices=["frontier", "visits", "discovery_time", "recent"],
        default="frontier",
        help="How to sort the states"
    )
    
    args = parser.parse_args()
    
    visualize_archive_states(
        args.archive_path,
        args.output_dir,
        args.max_states,
        args.sort_by
    )


if __name__ == "__main__":
    main()