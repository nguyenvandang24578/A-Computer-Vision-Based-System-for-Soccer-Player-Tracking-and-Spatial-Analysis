import json
import os
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np


class PositionTracker:
    """
    Class to track and save player positions throughout the match
    Records 2D field coordinates for each player by jersey number
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize position tracker
        
        Args:
            output_dir: Directory to save position data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Dictionary to store positions for each player
        # Format: {jersey_number: {"positions": [(x, y, frame)], "team": team_id}}
        self.player_positions = defaultdict(lambda: {"positions": [], "team": None})
        
        # Ball positions
        self.ball_positions = []  # [(x, y, frame)]
        
        # Frame counter
        self.current_frame = 0
        
        # Match statistics
        self.match_stats = {
            "total_frames": 0,
            "players_tracked": set(),
            "ball_detections": 0,
            "start_time": None,
            "end_time": None
        }
    
    def update_frame(self, frame_number: int):
        """Update current frame number"""
        self.current_frame = frame_number
        self.match_stats["total_frames"] = max(self.match_stats["total_frames"], frame_number)
    
    def add_player_position(self, jersey_number: str, x: float, y: float, team_id: int):
        """
        Add player position for current frame
        
        Args:
            jersey_number: Player's jersey number
            x, y: 2D field coordinates
            team_id: Team ID (0 or 1)
        """
        # Store position with frame number
        self.player_positions[jersey_number]["positions"].append((float(x), float(y), self.current_frame))
        
        # Update team info (in case it changes, though it shouldn't)
        if self.player_positions[jersey_number]["team"] is None:
            self.player_positions[jersey_number]["team"] = team_id
        
        # Add to tracked players
        self.match_stats["players_tracked"].add(jersey_number)
    
    def add_ball_position(self, x: float, y: float):
        """
        Add ball position for current frame
        
        Args:
            x, y: 2D field coordinates
        """
        self.ball_positions.append((float(x), float(y), self.current_frame))
        self.match_stats["ball_detections"] += 1
    
    def save_positions(self, filename: str = "match_positions.json"):
        """
        Save all tracked positions to JSON file with error handling
        
        Args:
            filename: Output filename
        """
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            # Convert defaultdict to regular dict and sets to lists for JSON serialization
            data = {
                "players": {},
                "ball": self.ball_positions,
                "match_stats": {
                    **self.match_stats,
                    "players_tracked": list(self.match_stats["players_tracked"]),
                    "total_frames": self.current_frame,
                    "data_integrity": "complete"
                }
            }
            
            # Safely convert player positions
            for jersey_number, player_data in self.player_positions.items():
                data["players"][str(jersey_number)] = {
                    "positions": player_data["positions"],
                    "team": player_data["team"]
                }
            
            # Validate data before saving
            if not self._validate_data(data):
                raise ValueError("Data validation failed")
            
            # Write to temporary file first
            temp_path = output_path + ".tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Move temp file to final location (atomic operation)
            import shutil
            shutil.move(temp_path, output_path)
            
            print(f"Position data saved successfully to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error saving position data: {e}")
            # Try to save a simplified version
            try:
                backup_path = output_path.replace('.json', '_backup.json')
                simplified_data = {
                    "players": dict(self.player_positions),
                    "ball": self.ball_positions[:1000],  # Limit ball positions
                    "match_stats": {
                        "total_frames": self.current_frame,
                        "error": str(e),
                        "players_tracked": list(self.match_stats["players_tracked"]) if hasattr(self.match_stats["players_tracked"], '__iter__') else []
                    }
                }
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(simplified_data, f, indent=2, ensure_ascii=False)
                print(f"Backup data saved to: {backup_path}")
                return backup_path
            except Exception as backup_error:
                print(f"Failed to save backup: {backup_error}")
                return None
    
    def _validate_data(self, data):
        """Validate data structure before saving"""
        try:
            # Check if data has required keys
            if not all(key in data for key in ["players", "ball", "match_stats"]):
                return False
            
            # Check players data
            for jersey_number, player_data in data["players"].items():
                if not isinstance(player_data, dict):
                    return False
                if "positions" not in player_data or "team" not in player_data:
                    return False
                if not isinstance(player_data["positions"], list):
                    return False
            
            # Check ball data
            if not isinstance(data["ball"], list):
                return False
            
            # Check match stats
            if not isinstance(data["match_stats"], dict):
                return False
            
            return True
            
        except Exception as e:
            print(f"Data validation error: {e}")
            return False
    
    def get_player_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all players"""
        summary = {}
        
        for jersey_number, data in self.player_positions.items():
            positions = data["positions"]
            if not positions:
                continue
            
            # Calculate basic statistics
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            frames = [pos[2] for pos in positions]
            
            summary[jersey_number] = {
                "team": data["team"],
                "total_detections": len(positions),
                "first_frame": min(frames),
                "last_frame": max(frames),
                "avg_position": (np.mean(x_coords), np.mean(y_coords)),
                "position_range": {
                    "x_min": min(x_coords), "x_max": max(x_coords),
                    "y_min": min(y_coords), "y_max": max(y_coords)
                },
                "distance_covered": self._calculate_distance_covered(positions)
            }
        
        return summary
    
    def _calculate_distance_covered(self, positions: List[Tuple[float, float, int]]) -> float:
        """Calculate total distance covered by a player"""
        if len(positions) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(positions)):
            x1, y1, _ = positions[i-1]
            x2, y2, _ = positions[i]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance += distance
        
        return total_distance
    
    def export_to_txt(self, filename: str = "match_positions.txt"):
        """
        Export positions to a simple text format
        Format: frame_number,jersey_number,team_id,x,y,object_type
        
        Args:
            filename: Output filename
        """
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("frame,jersey_number,team_id,x,y,object_type\n")
            
            # Write player positions
            for jersey_number, data in self.player_positions.items():
                team_id = data["team"]
                for x, y, frame in data["positions"]:
                    f.write(f"{frame},{jersey_number},{team_id},{x:.2f},{y:.2f},player\n")
            
            # Write ball positions
            for x, y, frame in self.ball_positions:
                f.write(f"{frame},ball,-1,{x:.2f},{y:.2f},ball\n")
        
        print(f"Position data exported to: {output_path}")
        return output_path
    
    def get_match_statistics(self) -> Dict[str, Any]:
        """Get comprehensive match statistics"""
        return {
            "total_frames": self.match_stats["total_frames"],
            "total_players": len(self.match_stats["players_tracked"]),
            "players_tracked": list(self.match_stats["players_tracked"]),
            "ball_detections": self.match_stats["ball_detections"],
            "average_players_per_frame": len(self.player_positions) / max(1, self.match_stats["total_frames"]),
            "ball_detection_rate": self.match_stats["ball_detections"] / max(1, self.match_stats["total_frames"])
        }

    @staticmethod
    def repair_json_file(broken_file_path: str, output_file_path: str = None):
        """
        Repair a broken JSON file by extracting valid data
        
        Args:
            broken_file_path: Path to the broken JSON file
            output_file_path: Path for the repaired file (optional)
        """
        if output_file_path is None:
            output_file_path = broken_file_path.replace('.json', '_repaired.json')
        
        print(f"🔧 Attempting to repair JSON file: {broken_file_path}")
        
        try:
            # Read the broken file as text
            with open(broken_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to find the last complete player entry
            import re
            
            # Extract players section
            players_match = re.search(r'"players":\s*{(.*?)}(?=\s*,\s*"ball"|\s*}$)', content, re.DOTALL)
            ball_match = re.search(r'"ball":\s*\[(.*?)\]', content, re.DOTALL)
            
            repaired_data = {
                "players": {},
                "ball": [],
                "match_stats": {
                    "total_frames": 0,
                    "players_tracked": [],
                    "ball_detections": 0,
                    "repaired": True,
                    "original_file": broken_file_path
                }
            }
            
            # Extract player data
            if players_match:
                players_content = players_match.group(1)
                # Find individual player entries
                player_pattern = r'"(\w+)":\s*{\s*"positions":\s*\[(.*?)\]\s*,\s*"team":\s*(\d+)\s*}'
                
                for match in re.finditer(player_pattern, players_content, re.DOTALL):
                    jersey_number = match.group(1)
                    positions_str = match.group(2)
                    team = int(match.group(3))
                    
                    # Parse positions
                    positions = []
                    position_pattern = r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*(\d+)\s*\]'
                    for pos_match in re.finditer(position_pattern, positions_str):
                        x = float(pos_match.group(1))
                        y = float(pos_match.group(2))
                        frame = int(pos_match.group(3))
                        positions.append([x, y, frame])
                    
                    if positions:  # Only add if we found valid positions
                        repaired_data["players"][jersey_number] = {
                            "positions": positions,
                            "team": team
                        }
                        repaired_data["match_stats"]["players_tracked"].append(jersey_number)
            
            # Extract ball data
            if ball_match:
                ball_content = ball_match.group(1)
                ball_pattern = r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*(\d+)\s*\]'
                
                for match in re.finditer(ball_pattern, ball_content):
                    x = float(match.group(1))
                    y = float(match.group(2))
                    frame = int(match.group(3))
                    repaired_data["ball"].append([x, y, frame])
            
            # Update match stats
            if repaired_data["players"]:
                all_frames = []
                for player_data in repaired_data["players"].values():
                    for pos in player_data["positions"]:
                        all_frames.append(pos[2])
                
                if all_frames:
                    repaired_data["match_stats"]["total_frames"] = max(all_frames)
            
            repaired_data["match_stats"]["ball_detections"] = len(repaired_data["ball"])
            
            # Save repaired data
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(repaired_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Repaired JSON saved to: {output_file_path}")
            print(f"   - Players recovered: {len(repaired_data['players'])}")
            print(f"   - Ball positions recovered: {len(repaired_data['ball'])}")
            print(f"   - Total frames: {repaired_data['match_stats']['total_frames']}")
            
            return output_file_path
            
        except Exception as e:
            print(f"❌ Failed to repair JSON file: {e}")
            return None
