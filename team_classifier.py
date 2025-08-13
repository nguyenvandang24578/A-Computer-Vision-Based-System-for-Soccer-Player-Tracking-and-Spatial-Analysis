import torch
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import cv2
from typing import List, Tuple, Optional
import pickle
import os

class FootballTeamClassifier:
    def __init__(self, device: str = "auto"):
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        self.is_fitted = False
        
        # Simple color-based team classification
        self.kmeans_model = None
        self.team_colors = None
        
    def extract_color_features(self, crop: np.ndarray) -> np.ndarray:
        """Extract enhanced color features from player crop focusing on jersey area"""
        if crop.size == 0:
            return np.zeros(18)  # Consistent 18-dimensional feature vector
        
        try:
            # Resize crop to standard size for consistency
            crop_resized = cv2.resize(crop, (64, 128))
            
            # Focus on jersey area (upper 60% of the crop)
            h, w = crop_resized.shape[:2]
            jersey_area = crop_resized[:int(h*0.6), :]
            
            # Convert to multiple color spaces for better discrimination
            hsv = cv2.cvtColor(jersey_area, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(jersey_area, cv2.COLOR_BGR2LAB)
            
            # HSV features (good for jersey colors) - 6 features total
            h_hist = cv2.calcHist([hsv], [0], None, [3], [0, 180])  # 3 bins for hue
            s_hist = cv2.calcHist([hsv], [1], None, [3], [0, 256])  # 3 bins for saturation
            
            # LAB features (good for color perception) - 6 features total  
            l_hist = cv2.calcHist([lab], [0], None, [3], [0, 256])  # 3 bins for lightness
            a_hist = cv2.calcHist([lab], [1], None, [3], [0, 256])  # 3 bins for a*
            b_hist = cv2.calcHist([lab], [2], None, [3], [0, 256])  # 3 bins for b*
            
            # Additional RGB features for completeness - 6 features total
            b_hist_rgb = cv2.calcHist([jersey_area], [0], None, [3], [0, 256])  # Blue
            g_hist_rgb = cv2.calcHist([jersey_area], [1], None, [3], [0, 256])  # Green  
            r_hist_rgb = cv2.calcHist([jersey_area], [2], None, [3], [0, 256])  # Red
            
            # Normalize histograms
            h_hist = h_hist.flatten() / (h_hist.sum() + 1e-7)
            s_hist = s_hist.flatten() / (s_hist.sum() + 1e-7)
            l_hist = l_hist.flatten() / (l_hist.sum() + 1e-7)
            a_hist = a_hist.flatten() / (a_hist.sum() + 1e-7)
            b_hist = b_hist.flatten() / (b_hist.sum() + 1e-7)
            b_hist_rgb = b_hist_rgb.flatten() / (b_hist_rgb.sum() + 1e-7)
            g_hist_rgb = g_hist_rgb.flatten() / (g_hist_rgb.sum() + 1e-7)
            r_hist_rgb = r_hist_rgb.flatten() / (r_hist_rgb.sum() + 1e-7)
            
            # Combine all features: 3+3+3+3+3+3 = 18 features
            features = np.concatenate([
                h_hist, s_hist,        # HSV: 6 features
                l_hist, a_hist, b_hist, # LAB: 9 features  
                b_hist_rgb, g_hist_rgb, r_hist_rgb  # RGB: 18 features
            ])
            
            return features
            
        except Exception as e:
            print(f"Error extracting color features: {e}")
            return np.zeros(18)
    
    def collect_training_data(self, video_path: str, model, stride: int = 30, max_crops: int = 500) -> List[np.ndarray]:
        """Collect player crops from video for training with better filtering"""
        print(f"Collecting training data from {video_path} with stride={stride}")
        
        frame_generator = sv.get_video_frames_generator(source_path=video_path, stride=stride)
        crops = []
        frame_count = 0
        
        for frame in tqdm(frame_generator, desc="Collecting crops"):
            if len(crops) >= max_crops:
                break
                
            result = model.predict(frame, conf=0.25, verbose=False)[0]  # Lower confidence
            
            xyxy = result.boxes.xyxy.cpu().numpy()
            confidence = result.boxes.conf.cpu().numpy()
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            
            detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
            detections = detections.with_nms(threshold=0.5, class_agnostic=True)
            
            # Debug: Print unique class IDs found
            if frame_count == 0 and len(detections) > 0:
                unique_classes = np.unique(detections.class_id)
                print(f"Debug: Found class IDs in first frame: {unique_classes}")
                print(f"Debug: Model class names: {model.names}")
            
            # Use class 0 for players (based on model output)
            player_detections = detections[detections.class_id == 0]
            
            # Filter by size to get good quality crops
            if len(player_detections) > 0:
                for xyxy in player_detections.xyxy:
                    w = xyxy[2] - xyxy[0]
                    h = xyxy[3] - xyxy[1]
                    if w > 30 and h > 60:  # Minimum size filter
                        crop = sv.crop_image(frame, xyxy)
                        if crop.size > 0:
                            crops.append(crop)
            
            frame_count += 1
            
        print(f"Collected {len(crops)} player crops from {frame_count} frames")
        return crops
    
    def fit(self, video_path: str, model, **kwargs):
        """Main training method using color-based KMeans clustering"""
        crops = self.collect_training_data(video_path, model, **kwargs)
        
        if len(crops) < 10:
            raise ValueError(f"Not enough training data. Only collected {len(crops)} crops")
        
        print(f"Extracting color features from {len(crops)} crops...")
        
        # Extract color features from all crops
        features = []
        for crop in tqdm(crops, desc="Extracting color features"):
            feature = self.extract_color_features(crop)
            features.append(feature)
        
        features = np.array(features)
        print(f"Feature shape: {features.shape}")
        
        # Try different numbers of clusters and pick the best
        best_score = -1
        best_kmeans = None
        
        # Only try 2 teams for football
        for n_clusters in [2]:  # Only 2 teams for football
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                
                if len(np.unique(labels)) > 1:  # Check if we have multiple clusters
                    score = silhouette_score(features, labels)
                    print(f"Clusters: {n_clusters}, Silhouette score: {score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_kmeans = kmeans
            except Exception as e:
                print(f"Failed clustering with {n_clusters} clusters: {e}")
                continue
        
        if best_kmeans is None:
            # Fallback to simple 2-cluster KMeans
            best_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            best_kmeans.fit(features)
            print("Using fallback 2-cluster solution")
        
        self.kmeans_model = best_kmeans
        labels = best_kmeans.labels_
        
        print(f"Clustering completed. Team distribution: {np.bincount(labels)}")
        self.is_fitted = True
        return True
    
    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """Predict team for given crops using color features"""
        if not self.is_fitted:
            raise ValueError("Classifier not fitted yet!")
        
        if len(crops) == 0:
            return np.array([])
        
        # Extract features and predict
        features = []
        for crop in crops:
            feature = self.extract_color_features(crop)
            features.append(feature)
        
        features = np.array(features)
        predictions = self.kmeans_model.predict(features)
        
        return predictions
    
    def save_classifier(self, path: str):
        """Save trained classifier"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_data = {
            'is_fitted': self.is_fitted,
            'kmeans_model': self.kmeans_model,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Classifier saved to {path}")
    
    def load_classifier(self, path: str):
        """Load trained classifier"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Classifier file {path} not found")
        
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.is_fitted = save_data['is_fitted']
        self.kmeans_model = save_data['kmeans_model']
        
        print(f"Classifier loaded from {path}")
