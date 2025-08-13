import cv2
import numpy as np
import json
from typing import List, Tuple, Dict, Any
from collections import defaultdict


class FieldMapper2D:
    """
    Class to create 2D field map visualization for football tracking
    Maps pixel coordinates from video to 2D field representation
    """
    
    def __init__(self, field_points_path: str = None, field_points: List[List[int]] = None, 
                 field_dimensions: Dict[str, float] = None, map_size: Tuple[int, int] = (400, 300)):
        """
        Initialize field mapper
        
        Args:
            field_points_path: Path to JSON file containing field corner points
            field_points: List of field corner points [top_left, top_right, bottom_left, bottom_right, center]
            field_dimensions: Dict with field length and width in meters
            map_size: Size of the 2D map (width, height)
        """
        self.map_size = map_size
        self.map_width, self.map_height = map_size
        
        # Default field dimensions for 7-a-side football
        self.field_dimensions = field_dimensions or {"length": 60, "width": 40}
        
        # Load field points
        if field_points_path:
            self.load_field_points(field_points_path)
        elif field_points:
            self.field_points = field_points
        else:
            raise ValueError("Either field_points_path or field_points must be provided")
        
        # Initialize transformation matrix
        self.transform_matrix = None
        self.inverse_transform_matrix = None
        self.setup_transformation()
        
        # Colors for teams and ball
        self.team_colors = {
            0: (255, 100, 100),  # Team 0 - Light Blue
            1: (255, 100, 255),  # Team 1 - Pink/Magenta
            'ball': (0, 255, 255),  # Ball - Yellow
            'field': (34, 139, 34),  # Field - Forest Green
            'lines': (255, 255, 255),  # Field lines - White
        }
        
        # Create base field image
        self.field_image = self.create_field_image()
        
    def load_field_points(self, file_path: str):
        """Load field points from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.field_points = data['field_points']
                if 'field_dimensions' in data:
                    self.field_dimensions = data['field_dimensions']
        except Exception as e:
            raise ValueError(f"Error loading field points: {e}")
    
    def setup_transformation(self):
        """Setup perspective transformation from video coordinates to 2D field"""
        if len(self.field_points) < 4:
            raise ValueError("At least 4 field points are required for transformation")
        
        # Video field corners: [top_left, top_right, bottom_left, bottom_right]
        src_points = np.array([
            self.field_points[0],  # top_left
            self.field_points[1],  # top_right  
            self.field_points[2],  # bottom_left
            self.field_points[3],  # bottom_right
        ], dtype=np.float32)
        
        # 2D map corners (with margin)
        margin = 20
        dst_points = np.array([
            [margin, margin],  # top_left
            [self.map_width - margin, margin],  # top_right
            [margin, self.map_height - margin],  # bottom_left
            [self.map_width - margin, self.map_height - margin],  # bottom_right
        ], dtype=np.float32)
        
        # Calculate transformation matrices
        self.transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inverse_transform_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
    
    def video_to_field_coords(self, video_coords: np.ndarray) -> np.ndarray:
        """Transform video coordinates to field coordinates"""
        if self.transform_matrix is None:
            return video_coords
        
        # Ensure input is in correct format
        if video_coords.ndim == 1:
            video_coords = video_coords.reshape(1, -1)
        
        # Add homogeneous coordinate if needed
        if video_coords.shape[1] == 2:
            ones = np.ones((video_coords.shape[0], 1))
            video_coords_h = np.hstack([video_coords, ones])
        else:
            video_coords_h = video_coords
        
        # Apply transformation
        field_coords_h = self.transform_matrix @ video_coords_h.T
        
        # Convert back to cartesian coordinates
        field_coords = field_coords_h[:2] / field_coords_h[2]
        
        return field_coords.T
    
    def field_to_video_coords(self, field_coords: np.ndarray) -> np.ndarray:
        """Transform field coordinates back to video coordinates"""
        if self.inverse_transform_matrix is None:
            return field_coords
        
        # Similar process but with inverse matrix
        if field_coords.ndim == 1:
            field_coords = field_coords.reshape(1, -1)
        
        if field_coords.shape[1] == 2:
            ones = np.ones((field_coords.shape[0], 1))
            field_coords_h = np.hstack([field_coords, ones])
        else:
            field_coords_h = field_coords
        
        video_coords_h = self.inverse_transform_matrix @ field_coords_h.T
        video_coords = video_coords_h[:2] / video_coords_h[2]
        
        return video_coords.T
    
    def create_field_image(self) -> np.ndarray:
        """Create base field image with lines and markings"""
        # Create green field
        field = np.full((self.map_height, self.map_width, 3), self.team_colors['field'], dtype=np.uint8)
        
        # Field boundaries
        margin = 20
        field_start_x, field_end_x = margin, self.map_width - margin
        field_start_y, field_end_y = margin, self.map_height - margin
        
        # Draw field boundary
        cv2.rectangle(field, (field_start_x, field_start_y), 
                     (field_end_x, field_end_y), self.team_colors['lines'], 2)
        
        # Center line
        center_x = self.map_width // 2
        cv2.line(field, (center_x, field_start_y), 
                (center_x, field_end_y), self.team_colors['lines'], 2)
        
        # Center circle
        center_y = self.map_height // 2
        circle_radius = min(self.map_width, self.map_height) // 8
        cv2.circle(field, (center_x, center_y), circle_radius, self.team_colors['lines'], 2)
        
        # Goal areas (simplified for 7-a-side)
        goal_width = (field_end_y - field_start_y) // 3
        goal_depth = (field_end_x - field_start_x) // 10
        
        # Left goal area
        cv2.rectangle(field, 
                     (field_start_x, center_y - goal_width//2),
                     (field_start_x + goal_depth, center_y + goal_width//2),
                     self.team_colors['lines'], 2)
        
        # Right goal area  
        cv2.rectangle(field,
                     (field_end_x - goal_depth, center_y - goal_width//2),
                     (field_end_x, center_y + goal_width//2),
                     self.team_colors['lines'], 2)
        
        # Penalty spots
        penalty_distance = goal_depth + 20
        cv2.circle(field, (field_start_x + penalty_distance, center_y), 3, self.team_colors['lines'], -1)
        cv2.circle(field, (field_end_x - penalty_distance, center_y), 3, self.team_colors['lines'], -1)
        
        return field
    
    def update_map(self, player_positions: List[Tuple[float, float]], 
                   player_teams: List[int], ball_position: Tuple[float, float] = None,
                   player_ids: List[str] = None) -> np.ndarray:
        """
        Update 2D map with current player and ball positions
        
        Args:
            player_positions: List of (x, y) coordinates in video space
            player_teams: List of team IDs for each player
            ball_position: (x, y) ball position in video space
            player_ids: List of player IDs/jersey numbers
            
        Returns:
            Updated field map image
        """
        # Start with clean field
        current_map = self.field_image.copy()
        
        # Transform player positions
        if player_positions:
            video_coords = np.array(player_positions)
            field_coords = self.video_to_field_coords(video_coords)
            
            # Draw players
            for i, ((fx, fy), team) in enumerate(zip(field_coords, player_teams)):
                fx, fy = int(fx), int(fy)
                
                # Skip if coordinates are out of bounds
                if fx < 0 or fx >= self.map_width or fy < 0 or fy >= self.map_height:
                    continue
                
                # Get team color
                color = self.team_colors.get(team, (128, 128, 128))
                
                # Draw player dot
                cv2.circle(current_map, (fx, fy), 6, color, -1)
                cv2.circle(current_map, (fx, fy), 6, (255, 255, 255), 1)
                
                # Draw player ID if available
                if player_ids and i < len(player_ids):
                    player_id = str(player_ids[i])
                    text_size = cv2.getTextSize(player_id, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    text_x = fx - text_size[0] // 2
                    text_y = fy + text_size[1] // 2
                    cv2.putText(current_map, player_id, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Transform and draw ball
        if ball_position:
            ball_video_coords = np.array([ball_position])
            ball_field_coords = self.video_to_field_coords(ball_video_coords)
            
            if len(ball_field_coords) > 0:
                bx, by = int(ball_field_coords[0][0]), int(ball_field_coords[0][1])
                
                # Check bounds
                if 0 <= bx < self.map_width and 0 <= by < self.map_height:
                    # Draw ball
                    cv2.circle(current_map, (bx, by), 4, self.team_colors['ball'], -1)
                    cv2.circle(current_map, (bx, by), 4, (0, 0, 0), 1)
        
        return current_map
    
    def get_field_stats(self, player_positions: List[Tuple[float, float]], 
                       player_teams: List[int]) -> Dict[str, Any]:
        """Get field statistics like team distribution, coverage etc."""
        if not player_positions:
            return {}
        
        # Transform positions
        video_coords = np.array(player_positions)
        field_coords = self.video_to_field_coords(video_coords)
        
        stats = {
            'total_players': len(player_positions),
            'team_counts': {},
            'field_coverage': {},
        }
        
        # Count players per team
        for team in player_teams:
            stats['team_counts'][team] = stats['team_counts'].get(team, 0) + 1
        
        # Calculate field coverage (simplified)
        if len(field_coords) > 0:
            x_coords = field_coords[:, 0]
            y_coords = field_coords[:, 1]
            
            stats['field_coverage'] = {
                'x_range': (float(np.min(x_coords)), float(np.max(x_coords))),
                'y_range': (float(np.min(y_coords)), float(np.max(y_coords))),
                'center_x': float(np.mean(x_coords)),
                'center_y': float(np.mean(y_coords)),
            }
        
        return stats
