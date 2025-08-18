"""Generate dashboard data for web viewer consumption."""

import json
import os
from typing import Dict, Any
from .event_logger import EventLogger


def generate_dashboard_data(event_log_dir: str, n_envs: int, output_file: str = None) -> Dict[str, Any]:
    """
    Generate dashboard data JSON file for web viewer.
    
    Args:
        event_log_dir: Directory containing event logs
        n_envs: Number of environments
        output_file: Optional output file path (default: dashboard.json in event_log_dir)
    """
    if not os.path.exists(event_log_dir):
        return {}
    
    # Use event logger to get aggregate stats
    logger = EventLogger(event_log_dir, n_envs)
    dashboard_data = logger.export_dashboard_data()
    logger.close()
    
    # Write to output file if specified
    if output_file is None:
        output_file = os.path.join(event_log_dir, "dashboard.json")
        
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
        print(f"Dashboard data written to {output_file}")
    except Exception as e:
        print(f"Failed to write dashboard data: {e}")
    
    return dashboard_data


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dashboard_generator.py <event_log_dir> [n_envs] [output_file]")
        sys.exit(1)
        
    event_log_dir = sys.argv[1]
    n_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    generate_dashboard_data(event_log_dir, n_envs, output_file)