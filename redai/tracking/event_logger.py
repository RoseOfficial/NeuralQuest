"""Event logging system for tracking game events across environments."""

import json
import csv
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from .pokemon_tracker import PokemonEvent, GameState


class EventLogger:
    """Logs Pokemon game events to files with minimal performance impact."""
    
    def __init__(self, log_dir: str, n_envs: int):
        """
        Initialize event logger.
        
        Args:
            log_dir: Directory to store event logs
            n_envs: Number of environments to track
        """
        self.log_dir = log_dir
        self.n_envs = n_envs
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Individual environment event files
        self.event_files = {}
        for env_id in range(n_envs):
            event_file = os.path.join(log_dir, f"env_{env_id:02d}_events.jsonl")
            self.event_files[env_id] = open(event_file, 'a', encoding='utf-8')
            
        # Aggregate statistics file
        self.stats_file = os.path.join(log_dir, "aggregate_stats.csv")
        self._init_stats_file()
        
        # Tracking counters
        self.event_counts = {env_id: {} for env_id in range(n_envs)}
        self.last_snapshot_time = datetime.now()
        
    def _init_stats_file(self):
        """Initialize CSV stats file with headers."""
        if not os.path.exists(self.stats_file):
            with open(self.stats_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'env_id', 'player_name', 'rival_name', 
                    'party_count', 'badges_count', 'money', 'current_map',
                    'total_events', 'pokemon_caught', 'badges_earned', 'locations_visited'
                ])
    
    def log_events(self, events: List[PokemonEvent]) -> None:
        """
        Log a batch of events.
        
        Args:
            events: List of events to log
        """
        if not events:
            return
            
        # Group events by environment
        events_by_env = {}
        for event in events:
            if event.env_id not in events_by_env:
                events_by_env[event.env_id] = []
            events_by_env[event.env_id].append(event)
            
        # Log events for each environment
        for env_id, env_events in events_by_env.items():
            self._log_env_events(env_id, env_events)
            
    def _log_env_events(self, env_id: int, events: List[PokemonEvent]) -> None:
        """Log events for a specific environment."""
        if env_id not in self.event_files:
            return
            
        file_handle = self.event_files[env_id]
        
        for event in events:
            # Convert event to JSON
            event_dict = {
                'timestamp': event.timestamp,
                'datetime': datetime.fromtimestamp(event.timestamp).isoformat(),
                'env_id': event.env_id,
                'event_type': event.event_type,
                'data': event.data
            }
            
            # Write to JSONL file
            json.dump(event_dict, file_handle, ensure_ascii=False)
            file_handle.write('\n')
            
            # Update counters
            if event.event_type not in self.event_counts[env_id]:
                self.event_counts[env_id][event.event_type] = 0
            self.event_counts[env_id][event.event_type] += 1
            
        # Flush to disk periodically
        file_handle.flush()
        
    def log_state_snapshot(self, env_id: int, state: GameState) -> None:
        """
        Log current game state snapshot to CSV.
        
        Args:
            env_id: Environment ID
            state: Current game state
        """
        timestamp = datetime.now()
        
        # Count total events for this environment
        total_events = sum(self.event_counts[env_id].values())
        
        with open(self.stats_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp.isoformat(),
                env_id,
                state.player_name,
                state.rival_name,
                state.party_count,
                len(state.badge_names),
                state.money,
                state.current_map,
                total_events,
                self.event_counts[env_id].get('pokemon_caught', 0),
                self.event_counts[env_id].get('badge_earned', 0),
                self.event_counts[env_id].get('location_changed', 0)
            ])
            
    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics across all environments."""
        stats = {
            'total_environments': self.n_envs,
            'total_events': 0,
            'events_by_type': {},
            'environments': {}
        }
        
        for env_id in range(self.n_envs):
            env_total = sum(self.event_counts[env_id].values())
            stats['total_events'] += env_total
            stats['environments'][env_id] = {
                'total_events': env_total,
                'events_by_type': self.event_counts[env_id].copy()
            }
            
            # Aggregate event types
            for event_type, count in self.event_counts[env_id].items():
                if event_type not in stats['events_by_type']:
                    stats['events_by_type'][event_type] = 0
                stats['events_by_type'][event_type] += count
                
        return stats
    
    def create_summary_report(self) -> str:
        """Create a human-readable summary report."""
        stats = self.get_aggregate_stats()
        
        report = f"""
Pokemon Training Event Summary
==============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Overall Statistics:
- Total Environments: {stats['total_environments']}
- Total Events: {stats['total_events']}

Event Breakdown:
"""
        
        for event_type, count in stats['events_by_type'].items():
            report += f"- {event_type.replace('_', ' ').title()}: {count}\n"
            
        report += "\nPer-Environment Summary:\n"
        
        for env_id in range(self.n_envs):
            env_stats = stats['environments'][env_id]
            if env_stats['total_events'] > 0:
                report += f"\nEnvironment {env_id:02d}: {env_stats['total_events']} events\n"
                for event_type, count in env_stats['events_by_type'].items():
                    if count > 0:
                        report += f"  - {event_type.replace('_', ' ').title()}: {count}\n"
                        
        return report
    
    def export_dashboard_data(self) -> Dict[str, Any]:
        """Export data formatted for web dashboard consumption."""
        current_states = {}
        recent_events = {}
        
        # Read recent events for each environment (last 10)
        for env_id in range(self.n_envs):
            event_file_path = os.path.join(self.log_dir, f"env_{env_id:02d}_events.jsonl")
            recent_events[env_id] = []
            
            if os.path.exists(event_file_path):
                try:
                    with open(event_file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        # Get last 10 lines
                        for line in lines[-10:]:
                            if line.strip():
                                event = json.loads(line.strip())
                                recent_events[env_id].append(event)
                except Exception:
                    pass  # Continue if file read fails
                    
        return {
            'stats': self.get_aggregate_stats(),
            'recent_events': recent_events,
            'timestamp': datetime.now().isoformat()
        }
        
    def close(self):
        """Close all file handles."""
        for file_handle in self.event_files.values():
            file_handle.close()
            
    def __del__(self):
        """Cleanup file handles on deletion."""
        try:
            self.close()
        except:
            pass