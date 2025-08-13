import cv2
import os
import argparse
import numpy as np
from ultralytics import YOLO
import supervision as sv
from team_classifier import FootballTeamClassifier
from ball_interpolator import BallInterpolator
from field_mapper_2d import FieldMapper2D
from position_tracker import PositionTracker
import time
from typing import Optional, Dict, Any, List, Tuple
import torch
from scipy.spatial.distance import cdist
from collections import defaultdict, Counter
from PIL import Image

class TeamAwareByteTracker:
    """Custom ByteTracker that uses team classification and jersey number recognition to improve tracking consistency"""
    
    def __init__(self, frame_rate: int = 30, track_thresh: float = 0.25, 
                 track_buffer: int = 30, match_thresh: float = 0.8, 
                 team_consistency_weight: float = 0.3, jersey_model_path: str = None):
        # Initialize ByteTracker with correct parameter names
        self.base_tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate,
            minimum_consecutive_frames=1
        )
        self.team_consistency_weight = team_consistency_weight
        self.tracker_team_history = {}  # {tracker_id: [team_predictions]}
        self.tracker_features = {}      # {tracker_id: recent_features}
        self.tracker_positions = {}     # {tracker_id: recent_positions}
        
        # Jersey number recognition
        self.jersey_model = None
        if jersey_model_path and os.path.exists(jersey_model_path):
            self.jersey_model = YOLO(jersey_model_path)
            if torch.cuda.is_available():
                self.jersey_model.to('cuda')
            print(f"Jersey number model loaded: {jersey_model_path}")
        
        # Jersey number tracking
        self.jersey_history = defaultdict(list)  # {tracker_id: [jersey_numbers]}
        self.jersey_to_tracker = {}              # {jersey_number: tracker_id}
        self.tracker_jersey_confidence = {}      # {tracker_id: confidence_score}
        
    def get_team_consistency_score(self, tracker_id: int, predicted_team: int) -> float:
        """Calculate team consistency score for a tracker"""
        if tracker_id not in self.tracker_team_history:
            return 0.5  # Neutral score for new trackers
        
        history = self.tracker_team_history[tracker_id]
        if len(history) == 0:
            return 0.5
        
        # Calculate percentage of consistent team predictions
        consistent_predictions = sum(1 for team in history if team == predicted_team)
        consistency_score = consistent_predictions / len(history)
        
        return consistency_score
    
    def update_team_history(self, tracker_id: int, team_id: int, max_history: int = 10):
        """Update team history for a tracker"""
        if tracker_id not in self.tracker_team_history:
            self.tracker_team_history[tracker_id] = []
        
        self.tracker_team_history[tracker_id].append(team_id)
        
        # Keep only recent history
        if len(self.tracker_team_history[tracker_id]) > max_history:
            self.tracker_team_history[tracker_id] = self.tracker_team_history[tracker_id][-max_history:]
    
    def extract_color_features(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract color features from bbox region"""
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Ensure bbox is within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return np.zeros(9)  # Return zero features for invalid ROI
        
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate color histogram features
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        
        # Normalize and flatten
        h_hist = h_hist.flatten() / (h_hist.sum() + 1e-7)
        s_hist = s_hist.flatten() / (s_hist.sum() + 1e-7)
        v_hist = v_hist.flatten() / (v_hist.sum() + 1e-7)
        
        # Take dominant colors (top 3 bins for each channel)
        h_dominant = np.sort(h_hist)[-3:]
        s_dominant = np.sort(s_hist)[-3:]
        v_dominant = np.sort(v_hist)[-3:]
        
        return np.concatenate([h_dominant, s_dominant, v_dominant])
    
    def calculate_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between color features"""
        if len(features1) == 0 or len(features2) == 0:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def calculate_position_consistency(self, tracker_id: int, current_bbox: np.ndarray) -> float:
        """Calculate position consistency for motion prediction"""
        if tracker_id not in self.tracker_positions:
            return 0.5
        
        positions = self.tracker_positions[tracker_id]
        if len(positions) < 2:
            return 0.5
        
        # Calculate center of current bbox
        current_center = np.array([(current_bbox[0] + current_bbox[2]) / 2,
                                  (current_bbox[1] + current_bbox[3]) / 2])
        
        # Calculate expected position based on recent movement
        if len(positions) >= 2:
            velocity = positions[-1] - positions[-2]
            expected_position = positions[-1] + velocity
            distance = np.linalg.norm(current_center - expected_position)
            
            # Normalize distance (smaller distance = higher consistency)
            max_distance = 100  # pixels
            consistency = max(0, 1 - distance / max_distance)
            return consistency
        
        return 0.5
    
    def update_position_history(self, tracker_id: int, bbox: np.ndarray, max_history: int = 5):
        """Update position history for a tracker"""
        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        
        if tracker_id not in self.tracker_positions:
            self.tracker_positions[tracker_id] = []
        
        self.tracker_positions[tracker_id].append(center)
        
        # Keep only recent history
        if len(self.tracker_positions[tracker_id]) > max_history:
            self.tracker_positions[tracker_id] = self.tracker_positions[tracker_id][-max_history:]
    
    def detect_jersey_number(self, frame: np.ndarray, bbox: np.ndarray, min_confidence: float = 0.6) -> str:
        """Detect jersey number from player crop with improved confidence"""
        if self.jersey_model is None:
            return None  # Return None when no detection, not "?"
        
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure bbox is within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # Crop using PIL (similar to notebook approach)
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cropped_pil = frame_pil.crop((x1, y1, x2, y2))
            cropped_np = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
            
            # Run jersey number detection with higher confidence threshold
            results_number = self.jersey_model(cropped_np, conf=min_confidence)[0]
            jersey_number = None
            
            if results_number.boxes:
                digits = []
                for b in results_number.boxes:
                    confidence = float(b.conf[0])
                    if confidence >= min_confidence:  # Additional confidence check
                        bx1, by1, bx2, by2 = map(int, b.xyxy[0])
                        cls_id = int(b.cls[0].item())
                        digit = self.jersey_model.names[cls_id]
                        
                        # Validate digit is numeric and in range 0-9
                        if digit.isdigit():
                            digits.append((bx1, digit, confidence))
                
                if digits:
                    # Sort digits by x-coordinate (left to right)
                    digits.sort(key=lambda tup: tup[0])
                    detected_number = ''.join([d[1] for d in digits])
                    
                    # Validate jersey number is in range 1-10
                    try:
                        number_int = int(detected_number)
                        if 1 <= number_int <= 10:
                            jersey_number = detected_number
                    except ValueError:
                        pass
            
            return jersey_number
            
        except Exception as e:
            print(f"Error detecting jersey number: {e}")
            return None
    
    def update_jersey_history(self, tracker_id: int, jersey_number: str, max_history: int = 15):
        """Update jersey number history for a tracker with improved persistence"""
        if jersey_number is not None:  # Only update when we have a valid detection
            self.jersey_history[tracker_id].append(jersey_number)
            
            # Keep only recent history
            if len(self.jersey_history[tracker_id]) > max_history:
                self.jersey_history[tracker_id] = self.jersey_history[tracker_id][-max_history:]
    
    def get_most_common_jersey(self, tracker_id: int) -> str:
        """Get most common jersey number for a tracker, maintaining previous number if no recent detections"""
        if tracker_id in self.jersey_history and self.jersey_history[tracker_id]:
            # Get the most common jersey number from history
            most_common = Counter(self.jersey_history[tracker_id]).most_common(1)[0][0]
            return most_common
        return None  # Return None if no history, will show as "?" in display
    
    def get_stable_jersey_number(self, tracker_id: int) -> str:
        """Get stable jersey number, keeping previous number if no new detection"""
        if not hasattr(self, 'last_known_jersey'):
            self.last_known_jersey = {}
        
        current_jersey = self.get_most_common_jersey(tracker_id)
        
        if current_jersey is not None:
            # Update last known jersey
            self.last_known_jersey[tracker_id] = current_jersey
            return current_jersey
        elif tracker_id in self.last_known_jersey:
            # Return last known jersey if no current detection
            return self.last_known_jersey[tracker_id]
        else:
            return "?"  # Only show "?" for completely new trackers with no history
    
    def get_jersey_confidence(self, tracker_id: int) -> float:
        """Calculate confidence score for jersey number based on consistency"""
        if tracker_id not in self.jersey_history or not self.jersey_history[tracker_id]:
            return 0.0
        
        history = self.jersey_history[tracker_id]
        if len(history) == 0:
            return 0.0
            
        most_common = Counter(history).most_common(1)[0]
        confidence = most_common[1] / len(history)  # Percentage of consistent readings
        
        # Boost confidence based on total detections
        detection_bonus = min(0.2, len(history) * 0.02)  # Up to 20% bonus for more detections
        
        return min(1.0, confidence + detection_bonus)
    
    def resolve_jersey_conflicts(self, detections: sv.Detections, current_jerseys: List[str]) -> List[int]:
        """Resolve conflicts when same jersey number is detected for multiple players with better conflict resolution"""
        corrected_tracker_ids = detections.tracker_id.copy()
        
        # Build jersey-to-trackers mapping for current frame (only for valid detections)
        jersey_to_current_trackers = defaultdict(list)
        for i, (tracker_id, jersey) in enumerate(zip(detections.tracker_id, current_jerseys)):
            if jersey is not None:  # Only consider valid jersey detections
                jersey_to_current_trackers[jersey].append((i, tracker_id))
        
        # Track assigned jerseys to prevent duplicates
        assigned_jerseys = set()
        
        # Resolve conflicts based on jersey history confidence
        for jersey, tracker_list in jersey_to_current_trackers.items():
            if len(tracker_list) > 1:  # Conflict detected
                # Find the tracker with highest jersey confidence for this number
                best_idx = -1
                best_confidence = -1
                
                for idx, tracker_id in tracker_list:
                    confidence = self.get_jersey_confidence(tracker_id)
                    # Bonus for trackers that historically had this jersey
                    historical_jersey = self.get_most_common_jersey(tracker_id)
                    if historical_jersey == jersey:
                        confidence += 0.5  # Boost confidence for historical match
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_idx = idx
                
                # Assign the jersey to the best match
                if best_idx != -1:
                    assigned_jerseys.add(jersey)
                    
                    # Try to reassign others to their historical jerseys or find alternatives
                    for idx, tracker_id in tracker_list:
                        if idx != best_idx:
                            # Try to assign historical jersey if available and not conflicting
                            historical_jersey = self.get_most_common_jersey(tracker_id)
                            if (historical_jersey is not None and 
                                historical_jersey not in assigned_jerseys and
                                historical_jersey != jersey):
                                # This tracker can keep its historical jersey
                                current_jerseys[idx] = historical_jersey
                                assigned_jerseys.add(historical_jersey)
                            else:
                                # Clear the conflicting jersey detection
                                current_jerseys[idx] = None
            else:
                # No conflict, mark jersey as assigned
                assigned_jerseys.add(jersey)
        
        return corrected_tracker_ids
    
    def find_best_jersey_match(self, current_tracker_id: int, detected_jersey: str, 
                              detections: sv.Detections) -> int:
        """Find best tracker match based on jersey history and position"""
        best_match = current_tracker_id
        best_score = 0
        
        # Look for trackers that historically had this jersey number
        for tracker_id in self.jersey_history:
            if tracker_id == current_tracker_id:
                continue
                
            # Check if this tracker historically had the detected jersey
            most_common_jersey = self.get_most_common_jersey(tracker_id)
            if most_common_jersey == detected_jersey:
                confidence = self.get_jersey_confidence(tracker_id)
                
                # Check if this tracker is currently active
                if tracker_id in detections.tracker_id:
                    tracker_idx = np.where(detections.tracker_id == tracker_id)[0]
                    if len(tracker_idx) > 0:
                        # Consider position consistency
                        position_score = self.calculate_position_consistency(tracker_id, 
                                                                           detections.xyxy[tracker_idx[0]])
                        
                        # Combined score
                        total_score = confidence * 0.7 + position_score * 0.3
                        
                        if total_score > best_score:
                            best_score = total_score
                            best_match = tracker_id
        
        return best_match
    
    def update_with_detections_and_teams(self, detections: sv.Detections, 
                                       frame: np.ndarray,
                                       team_predictions: List[int] = None) -> sv.Detections:
        """Enhanced update method - OPTIMIZED VERSION"""
        
        # First, get base tracking results
        tracked_detections = self.base_tracker.update_with_detections(detections)
        
        if len(tracked_detections) == 0:
            return tracked_detections
        
        # Extract current frame features and jerseys with optimized frequency
        current_features = []
        current_jerseys = []
        
        # Only process jersey detection every N frames for performance
        process_jerseys = hasattr(self, 'frame_counter') and (self.frame_counter % 5 == 0)
        if not hasattr(self, 'frame_counter'):
            self.frame_counter = 0
        self.frame_counter += 1
        
        for i, bbox in enumerate(tracked_detections.xyxy):
            # Always extract color features (they're fast)
            features = self.extract_color_features(frame, bbox)
            current_features.append(features)
            
            # Detect jersey number with reduced frequency
            if process_jerseys:
                jersey_number = self.detect_jersey_number(frame, bbox)
                current_jerseys.append(jersey_number)
            else:
                # Use cached jersey or None
                tracker_id = tracked_detections.tracker_id[i]
                cached_jersey = self.get_stable_jersey_number(tracker_id)
                current_jerseys.append(cached_jersey if cached_jersey != "?" else None)
        
        # Simplified jersey conflict resolution (only when processing jerseys)
        corrected_tracker_ids = tracked_detections.tracker_id.copy()
        if process_jerseys:
            corrected_tracker_ids = self.resolve_jersey_conflicts(tracked_detections, current_jerseys)
            tracked_detections.tracker_id = corrected_tracker_ids
        
        # Optimized team-aware correction (reduced frequency)
        if team_predictions is not None:
            for i, (tracker_id, predicted_team, jersey_number) in enumerate(zip(
                tracked_detections.tracker_id, team_predictions, current_jerseys)):
                
                # Update histories (lightweight operations)
                self.update_position_history(tracker_id, tracked_detections.xyxy[i])
                self.update_team_history(tracker_id, predicted_team)
                
                if process_jerseys:
                    self.update_jersey_history(tracker_id, jersey_number)
                
                # Store current features
                self.tracker_features[tracker_id] = current_features[i]
        
        return tracked_detections
    
    def attempt_id_correction(self, current_idx: int, current_tracker_id: int, 
                            current_team: int, current_features: np.ndarray,
                            tracked_detections: sv.Detections, team_predictions: List[int],
                            corrected_tracker_ids: np.ndarray) -> bool:
        """Attempt to correct ID switch based on team and feature similarity"""
        
        best_match_score = 0
        best_match_idx = -1
        
        # Look for better matches among other tracked objects
        for j, (other_tracker_id, other_team) in enumerate(zip(tracked_detections.tracker_id, team_predictions)):
            if j == current_idx:
                continue
            
            # Skip if not same team
            if other_team != current_team:
                continue
            
            # Calculate team consistency for the other tracker with current team
            other_team_consistency = self.get_team_consistency_score(other_tracker_id, current_team)
            
            # Calculate feature similarity
            if other_tracker_id in self.tracker_features:
                other_features = self.tracker_features[other_tracker_id]
                feature_similarity = self.calculate_feature_similarity(current_features, other_features)
            else:
                feature_similarity = 0.5
            
            # Calculate position consistency
            position_consistency = self.calculate_position_consistency(other_tracker_id, tracked_detections.xyxy[j])
            
            # Combined score: team consistency + feature similarity + position
            match_score = (other_team_consistency * 0.5 + 
                          feature_similarity * 0.3 + 
                          position_consistency * 0.2)
            
            if match_score > best_match_score and match_score > 0.7:  # Threshold for switching
                best_match_score = match_score
                best_match_idx = j
        
        # Perform ID switch if good match found
        if best_match_idx != -1:
            # Swap tracker IDs
            temp_id = corrected_tracker_ids[current_idx]
            corrected_tracker_ids[current_idx] = corrected_tracker_ids[best_match_idx]
            corrected_tracker_ids[best_match_idx] = temp_id
            
            print(f"ID correction: Swapped tracker {current_tracker_id} with {corrected_tracker_ids[best_match_idx]} (score: {best_match_score:.3f})")
            return True
        
        return False
    
    def attempt_jersey_aware_correction(self, current_idx: int, current_tracker_id: int,
                                      current_team: int, current_jersey: str, current_features: np.ndarray,
                                      tracked_detections: sv.Detections, team_predictions: List[int],
                                      current_jerseys: List[str], corrected_tracker_ids: np.ndarray) -> bool:
        """Attempt to correct ID switch using jersey numbers, team info, and features"""
        
        best_match_score = 0
        best_match_idx = -1
        
        # Look for better matches among other tracked objects
        for j, (other_tracker_id, other_team, other_jersey) in enumerate(
            zip(tracked_detections.tracker_id, team_predictions, current_jerseys)):
            
            if j == current_idx:
                continue
            
            # Skip if not same team
            if other_team != current_team:
                continue
            
            # Jersey matching score with improved logic
            jersey_score = 0.0
            if current_jersey is not None and other_jersey is not None:
                # Check if current jersey matches other tracker's history
                other_expected_jersey = self.get_most_common_jersey(other_tracker_id)
                current_expected_jersey = self.get_most_common_jersey(current_tracker_id)
                
                # Score based on jersey consistency
                if current_jersey == other_expected_jersey and other_jersey == current_expected_jersey:
                    jersey_score = 1.0  # Perfect swap match
                elif current_jersey == other_expected_jersey:
                    jersey_score = 0.8  # Good match for current
                elif other_jersey == current_expected_jersey:
                    jersey_score = 0.6  # Good match for other
                else:
                    jersey_score = 0.2  # No clear jersey advantage
            else:
                jersey_score = 0.3  # Neutral if jersey unclear
            
            # Team consistency score
            other_team_consistency = self.get_team_consistency_score(other_tracker_id, current_team)
            current_team_consistency = self.get_team_consistency_score(current_tracker_id, other_team)
            team_score = (other_team_consistency + current_team_consistency) / 2
            
            # Feature similarity
            if other_tracker_id in self.tracker_features:
                other_features = self.tracker_features[other_tracker_id]
                feature_similarity = self.calculate_feature_similarity(current_features, other_features)
            else:
                feature_similarity = 0.5
            
            # Position consistency
            position_consistency = self.calculate_position_consistency(other_tracker_id, tracked_detections.xyxy[j])
            
            # Combined score with higher weight on jersey matching
            match_score = (
                jersey_score * 0.5 +           # Jersey number matching is most important
                team_score * 0.25 +            # Team consistency
                feature_similarity * 0.15 +    # Color/appearance similarity  
                position_consistency * 0.1     # Position consistency
            )
            
            if match_score > best_match_score and match_score > 0.6:  # Threshold for switching
                best_match_score = match_score
                best_match_idx = j
        
        # Perform ID switch if good match found
        if best_match_idx != -1:
            # Swap tracker IDs
            temp_id = corrected_tracker_ids[current_idx]
            corrected_tracker_ids[current_idx] = corrected_tracker_ids[best_match_idx]
            corrected_tracker_ids[best_match_idx] = temp_id
            
            print(f"Jersey-aware ID correction: Swapped tracker {current_tracker_id} with {corrected_tracker_ids[best_match_idx]} "
                  f"(score: {best_match_score:.3f}, jersey: {current_jersey})")
            return True
        
        return False

class FootballTracker:
    def __init__(self, model_path: str, device: str = "auto", jersey_model_path: str = None, 
                 field_points_path: str = None, optimize_inference: bool = True):
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        self.model = YOLO(model_path)
        
        # Optimize model for inference
        if optimize_inference and self.device == "cuda":
            self.model.to(self.device)
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Use custom team-aware tracker with jersey recognition
        self.tracker = TeamAwareByteTracker(
            frame_rate=30,
            track_thresh=0.5,
            track_buffer=90,
            match_thresh=0.8,
            team_consistency_weight=0.3,
            jersey_model_path=jersey_model_path
        )
        self.team_classifier = FootballTeamClassifier(device=self.device)
        
        # Ball interpolator for missing ball detections
        self.ball_interpolator = BallInterpolator(max_history=50, max_extrapolation_gap=8)
        
        # Initialize 2D field mapper
        self.field_mapper = None
        if field_points_path and os.path.exists(field_points_path):
            try:
                self.field_mapper = FieldMapper2D(field_points_path=field_points_path)
                print(f"Field mapper initialized with: {field_points_path}")
            except Exception as e:
                print(f"Warning: Could not initialize field mapper: {e}")
        
        # Class IDs - Corrected based on model
        self.BALL_ID = 1  # ball
        self.PLAYER_ID = 0  # player
        
        # Setup annotators
        self.setup_annotators()
        
        # Tracking improvements
        self.team_history = {}  # Track team assignments over time
        self.confidence_threshold = 0.3  # Tăng threshold để giảm false positives
        self.nms_threshold = 0.5  # Tăng NMS threshold để loại bỏ nhiều overlap hơn
        
        # Performance optimizations
        self.frame_skip_team_classification = 3  # Chỉ chạy team classification mỗi 3 frames
        self.frame_skip_jersey_detection = 5     # Chỉ chạy jersey detection mỗi 5 frames
        self.max_detections = 20                 # Giới hạn số detection tối đa
        
        # Cache for expensive operations
        self.team_prediction_cache = {}
        self.jersey_detection_cache = {}
        
        # Frame tracking for ball interpolation
        self.current_frame_id = 0
        
        # Initialize position tracker for recording coordinates
        self.position_tracker = None
        
    def setup_annotators(self):
        """Setup supervision annotators with better styling"""
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493']),  # Only 2 colors for 2 teams
            thickness=2
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493']),  # Only 2 colors for 2 teams
            text_color=sv.Color.from_hex('#FFFFFF'),
            text_position=sv.Position.TOP_CENTER,  # Position at top for better visibility
            text_scale=0.8,  # Larger text
            text_thickness=2,
            text_padding=6  # More padding around text
        )
        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'),
            base=20,
            height=17,
            outline_thickness=2
        )
    
    def train_team_classifier(self, video_path: str, classifier_save_path: str = None):
        """Train team classifier on the video"""
        print("Training team classifier...")
        
        if classifier_save_path and os.path.exists(classifier_save_path):
            print(f"Loading existing classifier from {classifier_save_path}")
            self.team_classifier.load_classifier(classifier_save_path)
            return
        
        # Train new classifier
        self.team_classifier.fit(video_path, self.model, stride=30, max_crops=500)
        
        if classifier_save_path:
            self.team_classifier.save_classifier(classifier_save_path)
    
    def stabilize_team_assignment(self, tracker_id: int, predicted_team: int, window_size: int = 5) -> int:
        """Stabilize team assignment using temporal consistency"""
        if tracker_id not in self.team_history:
            self.team_history[tracker_id] = []
        
        self.team_history[tracker_id].append(predicted_team)
        
        # Keep only recent history
        if len(self.team_history[tracker_id]) > window_size:
            self.team_history[tracker_id] = self.team_history[tracker_id][-window_size:]
        
        # Return most common team in history
        history = self.team_history[tracker_id]
        return max(set(history), key=history.count)
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """Process a single frame and return detections and annotated frame - OPTIMIZED VERSION"""
        try:
            # Increment frame counter
            self.current_frame_id += 1
            
            # Resize frame for faster inference if it's too large
            original_shape = frame.shape[:2]
            if max(original_shape) > 1280:
                scale_factor = 1280 / max(original_shape)
                new_width = int(frame.shape[1] * scale_factor)
                new_height = int(frame.shape[0] * scale_factor)
                frame_resized = cv2.resize(frame, (new_width, new_height))
            else:
                frame_resized = frame
                scale_factor = 1.0
            
            # Run detection with optimized parameters
            with torch.no_grad():  # Disable gradient computation
                result = self.model.predict(
                    frame_resized, 
                    conf=self.confidence_threshold, 
                    verbose=False,
                    half=True if self.device == "cuda" else False,  # Use FP16 for speed
                    max_det=self.max_detections  # Limit detections
                )[0]
            
            # Convert to supervision format and scale back if needed
            xyxy = result.boxes.xyxy.cpu().numpy()
            if scale_factor != 1.0:
                xyxy = xyxy / scale_factor  # Scale coordinates back to original size
            
            confidence = result.boxes.conf.cpu().numpy()
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            
            detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        
        except Exception as e:
            print(f"Error in frame {self.current_frame_id} detection: {e}")
            # Return empty detections to continue processing
            return sv.Detections.empty(), sv.Detections.empty(), frame
        
        # Separate ball from other detections
        ball_detections = detections[detections.class_id == self.BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
        
        # Handle ball interpolation
        interpolated_ball = None
        if len(ball_detections) > 0:
            # Ball detected - add to interpolator history
            best_ball_idx = np.argmax(ball_detections.confidence)
            best_ball_bbox = ball_detections.xyxy[best_ball_idx]
            best_ball_conf = ball_detections.confidence[best_ball_idx]
            
            self.ball_interpolator.add_detection(
                self.current_frame_id, 
                best_ball_bbox, 
                best_ball_conf
            )
        else:
            # Ball not detected - request interpolation for this frame
            self.ball_interpolator.request_frame_interpolation(self.current_frame_id)
            
            # Try to get interpolated detection
            interpolated_ball = self.ball_interpolator.create_interpolated_detection(
                self.current_frame_id, 
                frame.shape
            )
            
            if interpolated_ball is not None:
                # Use interpolated ball detection
                ball_detections = interpolated_ball
                print(f"Frame {self.current_frame_id}: Using interpolated/extrapolated ball position")
        
        # Process other detections
        other_detections = detections[detections.class_id != self.BALL_ID]
        other_detections = other_detections.with_nms(threshold=self.nms_threshold, class_agnostic=True)
        
        # Separate by type before tracking
        players = other_detections[other_detections.class_id == self.PLAYER_ID]
        
        # STEP 1: Get preliminary team predictions with optimized frequency
        preliminary_team_predictions = None
        should_classify_teams = (self.current_frame_id % self.frame_skip_team_classification == 0)
        
        if len(players) > 0 and self.team_classifier.is_fitted and should_classify_teams:
            try:
                # Limit number of crops for performance
                max_crops = min(len(players), 12)  # Process max 12 players
                player_crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy[:max_crops]]
                preliminary_team_predictions = self.team_classifier.predict(player_crops)
                
                # Convert to list if it's numpy array
                if hasattr(preliminary_team_predictions, 'tolist'):
                    preliminary_team_predictions = preliminary_team_predictions.tolist()
                elif not isinstance(preliminary_team_predictions, list):
                    preliminary_team_predictions = list(preliminary_team_predictions)
                
                # Extend predictions if we had more players
                if len(players) > max_crops:
                    # Use cached predictions or default for remaining players
                    remaining_predictions = [0] * (len(players) - max_crops)
                    preliminary_team_predictions.extend(remaining_predictions)
            except Exception as e:
                print(f"Warning: Team classification failed at frame {self.current_frame_id}: {e}")
                preliminary_team_predictions = [0] * len(players)  # Default fallback
        elif len(players) > 0:
            # Use cached predictions or default values
            preliminary_team_predictions = [0] * len(players)  # Default team assignment
        
        # STEP 2: Use team-aware tracking for players with preliminary predictions
        tracked_players = sv.Detections.empty()
        if len(players) > 0:
            try:
                tracked_players = self.tracker.update_with_detections_and_teams(
                    players, frame, preliminary_team_predictions
                )
            except Exception as e:
                print(f"Warning: Tracking failed at frame {self.current_frame_id}: {e}")
                # Fallback to basic tracking
                tracked_players = self.tracker.base_tracker.update_with_detections(players)
        
        # STEP 3: Optimized final team predictions - only if needed
        if len(tracked_players) > 0 and self.team_classifier.is_fitted and should_classify_teams:
            try:
                # Create new crops from the FINAL tracked player positions (limited number)
                max_final_crops = min(len(tracked_players), 12)
                final_player_crops = [sv.crop_image(frame, xyxy) for xyxy in tracked_players.xyxy[:max_final_crops]]
                final_team_predictions = self.team_classifier.predict(final_player_crops)
                
                # Convert to list if it's numpy array
                if hasattr(final_team_predictions, 'tolist'):
                    final_team_predictions = final_team_predictions.tolist()
                elif not isinstance(final_team_predictions, list):
                    final_team_predictions = list(final_team_predictions)
                
                # Extend with cached or default predictions
                if len(tracked_players) > max_final_crops:
                    remaining_predictions = [0] * (len(tracked_players) - max_final_crops)
                    final_team_predictions.extend(remaining_predictions)
                
                # STEP 4: Apply stabilized team predictions to the tracked players
                stabilized_predictions = []
                for i, tracker_id in enumerate(tracked_players.tracker_id):
                    # Use the final team prediction (correctly aligned with tracked object)
                    team_pred = final_team_predictions[i] if i < len(final_team_predictions) else 0
                    stabilized_team = self.stabilize_team_assignment(
                        tracker_id, team_pred, window_size=5
                    )
                    stabilized_predictions.append(stabilized_team)
                
                # Assign final team colors
                tracked_players.class_id = np.array(stabilized_predictions)
                
            except Exception as e:
                print(f"Warning: Final team classification failed at frame {self.current_frame_id}: {e}")
                # Use previous team assignments or defaults
                stabilized_predictions = []
                for tracker_id in tracked_players.tracker_id:
                    if tracker_id in self.team_history and self.team_history[tracker_id]:
                        stabilized_team = self.team_history[tracker_id][-1]
                    else:
                        stabilized_team = 0
                    stabilized_predictions.append(stabilized_team)
                tracked_players.class_id = np.array(stabilized_predictions)
        elif len(tracked_players) > 0:
            # Use previous team assignments or defaults
            stabilized_predictions = []
            for tracker_id in tracked_players.tracker_id:
                if tracker_id in self.team_history and self.team_history[tracker_id]:
                    # Use most recent team assignment
                    stabilized_team = self.team_history[tracker_id][-1]
                else:
                    stabilized_team = 0  # Default team
                stabilized_predictions.append(stabilized_team)
            tracked_players.class_id = np.array(stabilized_predictions)
        
        # Merge all detections (only players now)
        all_tracked = tracked_players
        
        # Record positions to position tracker if enabled and field mapper available (optimized)
        if (self.position_tracker is not None and self.field_mapper is not None and 
            self.current_frame_id % 5 == 0):  # Only record every 5th frame for better performance
            
            self.position_tracker.update_frame(self.current_frame_id)
            
            # Record player positions (optimized batch processing)
            if len(all_tracked) > 0:
                # Filter valid detections first to reduce processing
                valid_detections = []
                valid_indices = []
                
                for i, bbox in enumerate(all_tracked.xyxy):
                    if i < len(all_tracked.tracker_id):
                        tracker_id = all_tracked.tracker_id[i]
                        jersey_number = self.tracker.get_stable_jersey_number(tracker_id)
                        
                        # Only process if we have a valid jersey number
                        if jersey_number != "?":
                            center_x = (bbox[0] + bbox[2]) / 2
                            center_y = (bbox[1] + bbox[3]) / 2
                            valid_detections.append((center_x, center_y))
                            valid_indices.append(i)
                
                # Batch transform only valid detections
                if valid_detections:
                    video_coords = np.array(valid_detections)
                    field_coords = self.field_mapper.video_to_field_coords(video_coords)
                    
                    # Record positions for valid transformations only
                    for idx, (field_x, field_y) in enumerate(field_coords):
                        original_idx = valid_indices[idx]
                        
                        # Quick bounds check
                        if (0 <= field_x <= self.field_mapper.map_width and 
                            0 <= field_y <= self.field_mapper.map_height):
                            
                            tracker_id = all_tracked.tracker_id[original_idx]
                            jersey_number = self.tracker.get_stable_jersey_number(tracker_id)
                            team_id = all_tracked.class_id[original_idx] if original_idx < len(all_tracked.class_id) else 0
                            
                            self.position_tracker.add_player_position(
                                jersey_number, field_x, field_y, team_id
                            )
            
            # Record ball position (optimized - less frequent for performance)
            if len(ball_detections) > 0 and self.current_frame_id % 3 == 0:  # Ball every 3rd frame
                ball_bbox = ball_detections.xyxy[0]
                ball_center_x = (ball_bbox[0] + ball_bbox[2]) / 2
                ball_center_y = (ball_bbox[1] + ball_bbox[3]) / 2
                
                # Quick transform for single point
                ball_video_coords = np.array([(ball_center_x, ball_center_y)])
                ball_field_coords = self.field_mapper.video_to_field_coords(ball_video_coords)
                
                if len(ball_field_coords) > 0:
                    ball_field_x, ball_field_y = ball_field_coords[0]
                    # Quick bounds check
                    if (0 <= ball_field_x <= self.field_mapper.map_width and 
                        0 <= ball_field_y <= self.field_mapper.map_height):
                        self.position_tracker.add_ball_position(ball_field_x, ball_field_y)
        
        return ball_detections, all_tracked, self.create_annotated_frame(frame, ball_detections, all_tracked, interpolated_ball is not None)
    
    def create_annotated_frame(self, frame: np.ndarray, ball_detections: sv.Detections, 
                             tracked_detections: sv.Detections, is_interpolated_ball: bool = False) -> np.ndarray:
        """Create annotated frame with jersey numbers only (no IDs) and interpolation indicator"""
        annotated_frame = frame.copy()
        
        # Annotate players with ellipses
        if len(tracked_detections) > 0:
            annotated_frame = self.ellipse_annotator.annotate(
                scene=annotated_frame,
                detections=tracked_detections
            )
            
            # Create labels with jersey numbers only (improved display)
            labels = []
            for tracker_id in tracked_detections.tracker_id:
                jersey_number = self.tracker.get_stable_jersey_number(tracker_id)
                labels.append(jersey_number)  # Will show stable number or "?" for new trackers
            
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame,
                detections=tracked_detections,
                labels=labels
            )
        
        # Annotate ball with triangle (different color if interpolated)
        if len(ball_detections) > 0:
            if is_interpolated_ball:
                # Use different color for interpolated ball
                interpolated_triangle_annotator = sv.TriangleAnnotator(
                    color=sv.Color.from_hex('#FF6B6B'),  # Red color for interpolated
                    base=20,
                    height=17,
                    outline_thickness=2
                )
                annotated_frame = interpolated_triangle_annotator.annotate(
                    scene=annotated_frame,
                    detections=ball_detections
                )
                
                # Add interpolation indicator text
                if len(ball_detections.xyxy) > 0:
                    ball_center = ball_detections.xyxy[0]
                    text_x = int((ball_center[0] + ball_center[2]) / 2)
                    text_y = int(ball_center[1] - 25)
                    cv2.putText(annotated_frame, "PRED", (text_x-15, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 107, 107), 2)
            else:
                # Normal ball annotation
                annotated_frame = self.triangle_annotator.annotate(
                    scene=annotated_frame,
                    detections=ball_detections
                )
        
        # Add 2D field map if field mapper is available
        if self.field_mapper is not None:
            annotated_frame = self.add_2d_map_overlay(annotated_frame, ball_detections, tracked_detections)
        
        return annotated_frame
    
    def add_2d_map_overlay(self, frame: np.ndarray, ball_detections: sv.Detections, 
                          tracked_detections: sv.Detections) -> np.ndarray:
        """Add 2D field map overlay to the bottom center of the frame"""
        try:
            # Prepare player data
            player_positions = []
            player_teams = []
            player_ids = []
            
            if len(tracked_detections) > 0:
                for i, bbox in enumerate(tracked_detections.xyxy):
                    # Get center of bounding box
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    player_positions.append((center_x, center_y))
                    
                    # Get team assignment
                    team_id = tracked_detections.class_id[i] if i < len(tracked_detections.class_id) else 0
                    player_teams.append(team_id)
                    
                    # Get jersey number
                    tracker_id = tracked_detections.tracker_id[i]
                    jersey_number = self.tracker.get_stable_jersey_number(tracker_id)
                    player_ids.append(jersey_number)
            
            # Prepare ball data
            ball_position = None
            if len(ball_detections) > 0 and len(ball_detections.xyxy) > 0:
                ball_bbox = ball_detections.xyxy[0]
                ball_center_x = (ball_bbox[0] + ball_bbox[2]) / 2
                ball_center_y = (ball_bbox[1] + ball_bbox[3]) / 2
                ball_position = (ball_center_x, ball_center_y)
            
            # Generate 2D map
            field_map = self.field_mapper.update_map(
                player_positions=player_positions,
                player_teams=player_teams,
                ball_position=ball_position,
                player_ids=player_ids
            )
            
            # Overlay map on frame
            frame_height, frame_width = frame.shape[:2]
            map_height, map_width = field_map.shape[:2]
            
            # Position map at bottom center
            margin_bottom = 20
            margin_horizontal = 20
            
            # Calculate position
            start_x = (frame_width - map_width) // 2
            start_y = frame_height - map_height - margin_bottom
            
            # Ensure map fits within frame
            if start_x < margin_horizontal:
                start_x = margin_horizontal
            elif start_x + map_width > frame_width - margin_horizontal:
                start_x = frame_width - map_width - margin_horizontal
            
            if start_y < 0:
                start_y = margin_bottom
            
            end_x = start_x + map_width
            end_y = start_y + map_height
            
            # Add semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (start_x-5, start_y-5), (end_x+5, end_y+5), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Overlay the map
            frame[start_y:end_y, start_x:end_x] = field_map
            
            # Add border around map
            cv2.rectangle(frame, (start_x-2, start_y-2), (end_x+2, end_y+2), (255, 255, 255), 2)
            
            # Add title
            title = "2D Field Map"
            title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            title_x = start_x + (map_width - title_size[0]) // 2
            title_y = start_y - 10
            cv2.putText(frame, title, (title_x, title_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        except Exception as e:
            print(f"Warning: Error creating 2D map overlay: {e}")
        
        return frame
    
    def process_video(self, video_path: str, output_path: str = None, 
                     skip_frames: int = 0, show_preview: bool = True,
                     train_classifier: bool = True, classifier_path: str = None,
                     save_positions: bool = True, positions_output_dir: str = None) -> Dict[str, Any]:
        """Process entire video"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} not found")
        
        # Initialize position tracker if requested and field mapper is available
        if save_positions and self.field_mapper is not None:
            if positions_output_dir is None:
                positions_output_dir = "output"
            self.position_tracker = PositionTracker(output_dir=positions_output_dir)
            print("Position tracking enabled")
        elif save_positions and self.field_mapper is None:
            print("Warning: Position tracking disabled - field mapper not available")
        
        # Train team classifier if needed
        if train_classifier:
            if classifier_path is None:
                classifier_path = "models/team_classifier.pkl"
            self.train_team_classifier(video_path, classifier_path)
        
        video_info = sv.VideoInfo.from_video_path(video_path)
        frame_generator = sv.get_video_frames_generator(source_path=video_path)
        
        # Setup output video if needed
        video_sink = None
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            video_sink = sv.VideoSink(target_path=output_path, video_info=video_info)
        
        # Setup preview window
        if show_preview:
            cv2.namedWindow('Football Tracker', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Football Tracker', 1280, 720)
        
        # Processing statistics
        stats = {
            'total_frames': video_info.total_frames,
            'processed_frames': 0,
            'detection_stats': {'ball': 0, 'players': 0},
            'processing_time': 0
        }
        
        frame_count = 0
        paused = False
        start_time = time.time()
        
        try:
            if video_sink:
                video_sink.__enter__()
            
            for frame in frame_generator:
                # Skip frames if needed
                if frame_count % (skip_frames + 1) != 0:
                    frame_count += 1
                    continue
                
                # Process frame
                ball_dets, tracked_dets, annotated_frame = self.process_frame(frame)
                
                # Update statistics
                stats['detection_stats']['ball'] += len(ball_dets)
                stats['detection_stats']['players'] += len(tracked_dets)
                
                # Get ball interpolation statistics
                ball_stats = self.ball_interpolator.get_statistics()
                
                # Add info overlay with interpolation info
                progress = (frame_count / video_info.total_frames) * 100
                info_text = f"Frame: {frame_count}/{video_info.total_frames} ({progress:.1f}%) | Players: {len(tracked_dets)} | Ball: {len(ball_dets)}"
                if ball_stats.get('interpolated_frames', 0) > 0:
                    info_text += f" | Interpolated: {ball_stats['interpolated_frames']}"
                
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add ball statistics if available
                if 'average_speed' in ball_stats:
                    speed_text = f"Ball Speed: {ball_stats['average_speed']:.1f} px/frame"
                    cv2.putText(annotated_frame, speed_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add detection info
                if ball_stats.get('total_detections', 0) > 0:
                    detection_text = f"Detections: {ball_stats['total_detections']} | Interpolations: {ball_stats['total_interpolations']} | Requested: {ball_stats.get('requested_frames', 0)}"
                    cv2.putText(annotated_frame, detection_text, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Handle preview
                if show_preview:
                    while paused:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('p'):
                            paused = False
                            break
                        elif key == ord('q'):
                            raise KeyboardInterrupt("User quit")
                    
                    cv2.imshow('Football Tracker', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        screenshot_path = f"screenshot_frame_{frame_count}.jpg"
                        cv2.imwrite(screenshot_path, annotated_frame)
                        print(f"Screenshot saved: {screenshot_path}")
                    elif key == ord('p'):
                        paused = True
                        print("Paused. Press 'p' again to resume.")
                
                # Write to output video
                if video_sink:
                    video_sink.write_frame(annotated_frame)
                
                frame_count += 1
                stats['processed_frames'] += 1
                
                # Print progress
                if stats['processed_frames'] % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = stats['processed_frames'] / elapsed
                    print(f"Progress: {progress:.1f}% | FPS: {fps:.1f} | Processed: {stats['processed_frames']}")
        
        except KeyboardInterrupt:
            print("Processing interrupted by user")
        
        finally:
            if video_sink:
                video_sink.__exit__(None, None, None)
            if show_preview:
                cv2.destroyAllWindows()
        
        stats['processing_time'] = time.time() - start_time
        
        # Save position data if position tracker was used
        if self.position_tracker is not None:
            # Save in both JSON and TXT formats
            json_path = self.position_tracker.save_positions("match_positions.json")
            txt_path = self.position_tracker.export_to_txt("match_positions.txt")
            
            # Get and add position statistics to stats
            position_stats = self.position_tracker.get_match_statistics()
            player_summary = self.position_tracker.get_player_summary()
            
            stats['position_tracking'] = {
                'json_file': json_path,
                'txt_file': txt_path,
                'match_stats': position_stats,
                'player_summary': player_summary
            }
            
            print(f"Position data saved:")
            print(f"  - JSON format: {json_path}")
            print(f"  - TXT format: {txt_path}")
            print(f"  - Total players tracked: {position_stats['total_players']}")
            print(f"  - Players: {', '.join(position_stats['players_tracked'])}")
        
        # Add ball interpolation statistics to final stats
        ball_interpolation_stats = self.ball_interpolator.get_statistics()
        stats['ball_interpolation'] = ball_interpolation_stats
        
        return stats

def main():
    parser = argparse.ArgumentParser(description="Enhanced Football Tracker with Team Classification and Jersey Recognition")
    parser.add_argument("--video_path", type=str, default="../input/Match_2031_5_0_test.mp4", help="Path to input video")
    parser.add_argument("--weights_path", type=str, default="../weight/best.pt", help="Path to YOLO weights")
    parser.add_argument("--jersey_model_path", type=str, default="../weight/jersey.pt", help="Path to jersey number recognition model")
    parser.add_argument("--field_points_path", type=str, default="../output/field_points.json", help="Path to field points JSON file")
    parser.add_argument("--output_path", type=str, default="../output/tracked_output.mp4", help="Path to output video")
    parser.add_argument("--skip_frames", type=int, default=0, help="Frames to skip")
    parser.add_argument("--no_preview", action="store_true", help="Disable preview")
    parser.add_argument("--classifier_path", type=str, default="../models/team_classifier.pkl", 
                       help="Path to save/load team classifier")
    parser.add_argument("--no_train", action="store_true", help="Skip team classifier training")
    parser.add_argument("--save_positions", action="store_true", help="Save player and ball positions to file")
    parser.add_argument("--positions_output_dir", type=str, default="../output", 
                       help="Directory to save position tracking files")
    parser.add_argument("--fast_mode", action="store_true", help="Enable fast processing mode (lower quality)")
    parser.add_argument("--max_detections", type=int, default=20, help="Maximum number of detections per frame")
    parser.add_argument("--team_classification_skip", type=int, default=3, 
                       help="Skip team classification every N frames (1=every frame, 3=every 3rd frame)")
    parser.add_argument("--jersey_detection_skip", type=int, default=5,
                       help="Skip jersey detection every N frames")
    
    args = parser.parse_args()
    
    # Create output and models directories if they don't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.classifier_path), exist_ok=True)
    
    # Check if required files exist
    required_files = [args.video_path, args.weights_path]
    if args.jersey_model_path:
        required_files.append(args.jersey_model_path)
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
            print(f"Current working directory: {os.getcwd()}")
            print("Please check the file paths or run from the correct directory.")
            return
    
    # Initialize tracker with jersey model and field mapper
    tracker = FootballTracker(
        model_path=args.weights_path, 
        jersey_model_path=args.jersey_model_path,
        field_points_path=args.field_points_path,
        optimize_inference=True
    )
    
    # Apply fast mode optimizations if requested
    if args.fast_mode:
        tracker.confidence_threshold = 0.4  # Higher threshold for faster inference
        tracker.nms_threshold = 0.6
        tracker.max_detections = args.max_detections
        tracker.frame_skip_team_classification = args.team_classification_skip
        tracker.frame_skip_jersey_detection = args.jersey_detection_skip
        print("Fast mode enabled - some accuracy may be reduced for better performance")
    
    # Process video
    stats = tracker.process_video(
        video_path=args.video_path,
        output_path=args.output_path,
        skip_frames=args.skip_frames,
        show_preview=not args.no_preview,
        train_classifier=not args.no_train,
        classifier_path=args.classifier_path,
        save_positions=args.save_positions,
        positions_output_dir=args.positions_output_dir
    )
    
    # Print final statistics
    print("\n" + "="*50)
    print("PROCESSING COMPLETED")
    print("="*50)
    print(f"Total frames processed: {stats['processed_frames']}/{stats['total_frames']}")
    print(f"Processing time: {stats['processing_time']:.2f}s")
    print(f"Average FPS: {stats['processed_frames']/stats['processing_time']:.2f}")
    print(f"Detection stats: {stats['detection_stats']}")
    
    # Print ball interpolation statistics
    if 'ball_interpolation' in stats:
        ball_stats = stats['ball_interpolation']
        print(f"Ball interpolation stats:")
        print(f"  - Total detections: {ball_stats.get('total_detections', 0)}")
        print(f"  - Total interpolations: {ball_stats.get('total_interpolations', 0)}")
        print(f"  - Current detections in memory: {ball_stats.get('current_detections', 0)}")
        print(f"  - Active interpolated frames: {ball_stats.get('interpolated_frames', 0)}")
        print(f"  - Total requested frames: {ball_stats.get('requested_frames', 0)}")
        if 'average_speed' in ball_stats:
            print(f"  - Average ball speed: {ball_stats['average_speed']:.2f} px/frame")
        if 'detection_range' in ball_stats:
            range_info = ball_stats['detection_range']
            print(f"  - Detection range: frames {range_info['min_frame']}-{range_info['max_frame']}")
        if 'average_confidence' in ball_stats:
            print(f"  - Average detection confidence: {ball_stats['average_confidence']:.3f}")
    
    # Print position tracking statistics
    if 'position_tracking' in stats:
        pos_stats = stats['position_tracking']
        print(f"Position tracking stats:")
        print(f"  - Total players tracked: {pos_stats['match_stats']['total_players']}")
        print(f"  - Players: {', '.join(pos_stats['match_stats']['players_tracked'])}")
        print(f"  - Ball detection rate: {pos_stats['match_stats']['ball_detection_rate']:.2%}")
        print(f"  - Data files saved:")
        print(f"    * JSON: {pos_stats['json_file']}")
        print(f"    * TXT: {pos_stats['txt_file']}")

if __name__ == "__main__":
    main()