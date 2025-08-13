import numpy as np
from typing import List, Tuple, Optional, Dict
import cv2
from collections import deque
import pandas as pd
import supervision as sv

class BallInterpolator:
    """Class for interpolating ball positions when detection is missing using pandas DataFrame"""
    
    def __init__(self, max_history: int = 50, max_extrapolation_gap: int = 5):
        """
        Initialize ball interpolator
        
        Args:
            max_history: Maximum number of frames to keep in history
            max_extrapolation_gap: Maximum frames to extrapolate beyond last detection
        """
        self.max_history = max_history
        self.max_extrapolation_gap = max_extrapolation_gap
        
        # Ball tracking data - store as list of dictionaries for easy DataFrame conversion
        self.ball_positions = []  # [{'frame_id': int, 'x1': float, 'y1': float, 'x2': float, 'y2': float, 'confidence': float}]
        self.interpolated_positions = {}  # {frame_id: bbox}
        
        # Statistics
        self.total_detections = 0
        self.total_interpolations = 0
        self.requested_frames = set()  # Track which frames were requested for interpolation
        
    def add_detection(self, frame_id: int, bbox: np.ndarray, confidence: float):
        """Add a ball detection to history"""
        x1, y1, x2, y2 = bbox
        
        # Add to ball positions list
        ball_data = {
            'frame_id': frame_id,
            'x1': float(x1),
            'y1': float(y1), 
            'x2': float(x2),
            'y2': float(y2),
            'confidence': float(confidence)
        }
        
        self.ball_positions.append(ball_data)
        self.total_detections += 1
        
        # Keep only recent history
        if len(self.ball_positions) > self.max_history:
            self.ball_positions = self.ball_positions[-self.max_history:]
        
        # Perform interpolation for all requested frames
        self._interpolate_all_requested_frames()
    
    def _interpolate_all_requested_frames(self):
        """Interpolate all requested frames using current detection data"""
        if len(self.ball_positions) < 1:
            return
        
        # Clear previous interpolations
        self.interpolated_positions = {}
        
        # If we only have one detection, we can't interpolate much
        if len(self.ball_positions) == 1:
            return
        
        # Convert ball positions to DataFrame
        df = pd.DataFrame(self.ball_positions)
        df = df.sort_values('frame_id')
        
        # Get detection range
        min_detected_frame = df['frame_id'].min()
        max_detected_frame = df['frame_id'].max()
        
        # Determine interpolation range based on requested frames
        interpolation_frames = []
        
        for requested_frame in self.requested_frames:
            # Include frames between detections (interpolation)
            if min_detected_frame <= requested_frame <= max_detected_frame:
                interpolation_frames.append(requested_frame)
            # Include frames near detections (extrapolation)
            elif (requested_frame < min_detected_frame and 
                  min_detected_frame - requested_frame <= self.max_extrapolation_gap):
                interpolation_frames.append(requested_frame)
            elif (requested_frame > max_detected_frame and 
                  requested_frame - max_detected_frame <= self.max_extrapolation_gap):
                interpolation_frames.append(requested_frame)
        
        if not interpolation_frames:
            return
        
        # Create extended frame range for interpolation/extrapolation
        all_frames = sorted(list(set(df['frame_id'].tolist() + interpolation_frames)))
        
        # Create complete DataFrame
        complete_df = pd.DataFrame({'frame_id': all_frames})
        complete_df = complete_df.merge(df, on='frame_id', how='left')
        complete_df = complete_df.set_index('frame_id')
        
        # For extrapolation beyond the range, we need special handling
        if len(df) >= 2:
            # Calculate velocity from last two points for forward extrapolation
            last_two = df.tail(2)
            if len(last_two) == 2:
                dt = last_two.iloc[1]['frame_id'] - last_two.iloc[0]['frame_id']
                if dt > 0:
                    vx1 = (last_two.iloc[1]['x1'] - last_two.iloc[0]['x1']) / dt
                    vy1 = (last_two.iloc[1]['y1'] - last_two.iloc[0]['y1']) / dt
                    vx2 = (last_two.iloc[1]['x2'] - last_two.iloc[0]['x2']) / dt
                    vy2 = (last_two.iloc[1]['y2'] - last_two.iloc[0]['y2']) / dt
                    
                    # Extrapolate forward
                    for frame_id in interpolation_frames:
                        if frame_id > max_detected_frame:
                            gap = frame_id - max_detected_frame
                            last_detection = df.iloc[-1]
                            
                            # Apply simple linear extrapolation with slight deceleration
                            decel_factor = 0.95 ** gap  # Slight deceleration over time
                            
                            extrapolated_x1 = last_detection['x1'] + vx1 * gap * decel_factor
                            extrapolated_y1 = last_detection['y1'] + vy1 * gap * decel_factor
                            extrapolated_x2 = last_detection['x2'] + vx2 * gap * decel_factor
                            extrapolated_y2 = last_detection['y2'] + vy2 * gap * decel_factor
                            
                            complete_df.loc[frame_id, 'x1'] = extrapolated_x1
                            complete_df.loc[frame_id, 'y1'] = extrapolated_y1
                            complete_df.loc[frame_id, 'x2'] = extrapolated_x2
                            complete_df.loc[frame_id, 'y2'] = extrapolated_y2
                            complete_df.loc[frame_id, 'confidence'] = 0.3  # Lower confidence for extrapolation
            
            # Calculate velocity from first two points for backward extrapolation
            first_two = df.head(2)
            if len(first_two) == 2:
                dt = first_two.iloc[1]['frame_id'] - first_two.iloc[0]['frame_id']
                if dt > 0:
                    vx1 = (first_two.iloc[1]['x1'] - first_two.iloc[0]['x1']) / dt
                    vy1 = (first_two.iloc[1]['y1'] - first_two.iloc[0]['y1']) / dt
                    vx2 = (first_two.iloc[1]['x2'] - first_two.iloc[0]['x2']) / dt
                    vy2 = (first_two.iloc[1]['y2'] - first_two.iloc[0]['y2']) / dt
                    
                    # Extrapolate backward
                    for frame_id in interpolation_frames:
                        if frame_id < min_detected_frame:
                            gap = min_detected_frame - frame_id
                            first_detection = df.iloc[0]
                            
                            # Apply simple linear extrapolation backward
                            decel_factor = 0.95 ** gap  # Slight deceleration over time
                            
                            extrapolated_x1 = first_detection['x1'] - vx1 * gap * decel_factor
                            extrapolated_y1 = first_detection['y1'] - vy1 * gap * decel_factor
                            extrapolated_x2 = first_detection['x2'] - vx2 * gap * decel_factor
                            extrapolated_y2 = first_detection['y2'] - vy2 * gap * decel_factor
                            
                            complete_df.loc[frame_id, 'x1'] = extrapolated_x1
                            complete_df.loc[frame_id, 'y1'] = extrapolated_y1
                            complete_df.loc[frame_id, 'x2'] = extrapolated_x2
                            complete_df.loc[frame_id, 'y2'] = extrapolated_y2
                            complete_df.loc[frame_id, 'confidence'] = 0.3  # Lower confidence for extrapolation
        
        # Interpolate missing values using linear interpolation
        interpolated_df = complete_df.interpolate(method='linear')
        
        # Store interpolated positions for missing frames
        for frame_id in interpolation_frames:
            if frame_id not in df['frame_id'].values:  # Only store if not actual detection
                if frame_id in interpolated_df.index and not interpolated_df.loc[frame_id].isna().any():
                    row = interpolated_df.loc[frame_id]
                    bbox = np.array([row['x1'], row['y1'], row['x2'], row['y2']])
                    self.interpolated_positions[frame_id] = bbox
                    self.total_interpolations += 1
        
        print(f"Interpolated/extrapolated {len(self.interpolated_positions)} ball positions for requested frames")
    
    def request_frame_interpolation(self, frame_id: int):
        """Request interpolation for a specific frame"""
        self.requested_frames.add(frame_id)
        
        # Try to interpolate immediately if we have enough data
        if len(self.ball_positions) >= 1:
            self._interpolate_all_requested_frames()
    
    def _interpolate_missing_positions(self):
        """Legacy method - kept for compatibility"""
        pass  # Replaced by _interpolate_all_requested_frames()
    
    def get_interpolated_position(self, frame_id: int) -> Optional[np.ndarray]:
        """Get interpolated position for a specific frame"""
        return self.interpolated_positions.get(frame_id)
    
    def create_interpolated_detection(self, frame_id: int, frame_shape: Tuple[int, int]) -> Optional[sv.Detections]:
        """Create a detection object for interpolated ball position"""
        bbox = self.get_interpolated_position(frame_id)
        
        if bbox is None:
            return None
        
        # Ensure position is within frame bounds
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        
        # Create detection
        xyxy = np.array([[x1, y1, x2, y2]])
        confidence = np.array([0.6])  # Medium confidence for interpolated
        class_id = np.array([1])  # Ball class ID
        
        detection = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )
        
        return detection
    
    def has_detections(self) -> bool:
        """Check if we have any ball detections"""
        return len(self.ball_positions) > 0
    
    def get_latest_detection_frame(self) -> Optional[int]:
        """Get the frame ID of the latest detection"""
        if not self.ball_positions:
            return None
        return max(pos['frame_id'] for pos in self.ball_positions)
    
    def get_earliest_detection_frame(self) -> Optional[int]:
        """Get the frame ID of the earliest detection"""
        if not self.ball_positions:
            return None
        return min(pos['frame_id'] for pos in self.ball_positions)
    
    def reset(self):
        """Reset the interpolator state"""
        self.ball_positions.clear()
        self.interpolated_positions.clear()
        self.requested_frames.clear()
        self.total_detections = 0
        self.total_interpolations = 0
    
    def get_statistics(self) -> Dict:
        """Get interpolator statistics"""
        stats = {
            'total_detections': self.total_detections,
            'total_interpolations': self.total_interpolations,
            'current_detections': len(self.ball_positions),
            'interpolated_frames': len(self.interpolated_positions),
            'requested_frames': len(self.requested_frames)
        }
        
        if self.ball_positions:
            stats['detection_range'] = {
                'min_frame': self.get_earliest_detection_frame(),
                'max_frame': self.get_latest_detection_frame()
            }
            
            # Calculate average confidence
            avg_confidence = np.mean([pos['confidence'] for pos in self.ball_positions])
            stats['average_confidence'] = avg_confidence
            
            # Calculate ball movement speed (pixels per frame)
            if len(self.ball_positions) >= 2:
                # Sort positions by frame_id
                sorted_positions = sorted(self.ball_positions, key=lambda x: x['frame_id'])
                
                # Calculate center points and distances
                distances = []
                for i in range(1, len(sorted_positions)):
                    prev_pos = sorted_positions[i-1]
                    curr_pos = sorted_positions[i]
                    
                    prev_center_x = (prev_pos['x1'] + prev_pos['x2']) / 2
                    prev_center_y = (prev_pos['y1'] + prev_pos['y2']) / 2
                    curr_center_x = (curr_pos['x1'] + curr_pos['x2']) / 2
                    curr_center_y = (curr_pos['y1'] + curr_pos['y2']) / 2
                    
                    distance = np.sqrt((curr_center_x - prev_center_x)**2 + (curr_center_y - prev_center_y)**2)
                    frame_gap = curr_pos['frame_id'] - prev_pos['frame_id']
                    
                    if frame_gap > 0:
                        speed = distance / frame_gap
                        distances.append(speed)
                
                if distances:
                    stats['average_speed'] = np.mean(distances)
                    stats['max_speed'] = np.max(distances)
        
        return stats
