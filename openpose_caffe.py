import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from typing import List, Tuple, Optional

class OpenPoseCaffe:
    """OpenPose implementation using Caffe models"""
    
    def __init__(self, model_dir: str = "openpose/models/"):
        self.model_dir = model_dir
        
        # Try different models in order of preference
        self.models_to_try = [
            ("pose/body_25/pose_deploy.prototxt", 
             "pose/body_25/pose_iter_584000.caffemodel", 25),
            ("pose/coco/pose_deploy_linevec.prototxt", 
             "pose/coco/pose_iter_440000.caffemodel", 18),
            ("pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt", 
             "pose/mpi/pose_iter_160000.caffemodel", 15)
        ]
        
        self.net = None
        self.num_points = 0
        self.pose_pairs = []
        
        self._load_model()
    
    def _load_model(self):
        """Load the first available OpenPose model"""
        for prototxt_rel, caffemodel_rel, num_points in self.models_to_try:
            prototxt = os.path.join(self.model_dir, prototxt_rel)
            caffemodel = os.path.join(self.model_dir, caffemodel_rel)
            
            if os.path.exists(prototxt) and os.path.exists(caffemodel):
                print(f"Loading model: {prototxt_rel}")
                self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
                self.num_points = num_points
                
                # Set up connections based on model
                self._setup_pose_pairs()
                
                # Try to use GPU
                try:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    print("Using CUDA backend")
                except:
                    print("Using CPU backend")
                
                return
        
        raise FileNotFoundError("No OpenPose model found in the specified directory")
    
    def _setup_pose_pairs(self):
        """Setup pose connections based on model type"""
        if self.num_points == 25:  # BODY_25
            self.pose_pairs = [
                (0,1), (1,2), (2,3), (3,4),
                (1,5), (5,6), (6,7),
                (1,8), (8,9), (9,10), (10,11),
                (8,12), (12,13), (13,14),
                (0,15), (15,17),
                (0,16), (16,18),
                (11,22), (22,23), (11,24),
                (14,19), (19,20), (14,21)
            ]
        elif self.num_points == 18:  # COCO
            self.pose_pairs = [
                [1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
                [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
                [1,0], [0,14], [14,16], [0,15], [15,17]
            ]
        elif self.num_points == 15:  # MPI
            self.pose_pairs = [
                [0,1], [1,2], [2,3], [3,4], [1,5], [5,6],
                [6,7], [1,14], [14,8], [8,9], [9,10], [14,11],
                [11,12], [12,13]
            ]
    
    def process_image(self, image: np.ndarray, threshold: float = 0.1) -> Tuple[List, np.ndarray]:
        """
        Process image and return keypoints + annotated image
        """
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Prepare blob
        net_input_size = (368, 368)
        blob = cv2.dnn.blobFromImage(
            image, 
            1.0 / 255, 
            net_input_size,
            (0, 0, 0), 
            swapRB=False, 
            crop=False
        )
        
        # Forward pass
        self.net.setInput(blob)
        output = self.net.forward()
        
        # Extract keypoints
        points = []
        for i in range(self.num_points):
            prob_map = output[0, i, :, :]
            _, prob, _, point = cv2.minMaxLoc(prob_map)
            
            # Scale point to original image
            x = (w * point[0]) / output.shape[3]
            y = (h * point[1]) / output.shape[2]
            
            if prob > threshold:
                points.append((float(x), float(y), float(prob)))
            else:
                points.append((0.0, 0.0, 0.0))
        
        # Draw on image
        output_image = self._draw_skeleton(image.copy(), points)
        
        return points, output_image
    
    def _draw_skeleton(self, image: np.ndarray, points: List) -> np.ndarray:
        """Draw skeleton on image"""
        for pair in self.pose_pairs:
            part_a = pair[0]
            part_b = pair[1]
            
            if (points[part_a][2] > 0.1 and points[part_b][2] > 0.1):
                pt1 = (int(points[part_a][0]), int(points[part_a][1]))
                pt2 = (int(points[part_b][0]), int(points[part_b][1]))
                cv2.line(image, pt1, pt2, (0, 255, 255), 2)
                cv2.circle(image, pt1, 5, (0, 0, 255), -1)
                cv2.circle(image, pt2, 5, (0, 0, 255), -1)
        
        # Draw all points
        for i, (x, y, conf) in enumerate(points):
            if conf > 0.1:
                cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), -1)
        
        return image
    
    def extract_features(self, image: np.ndarray) -> torch.Tensor:
        """
        Extract pose features as tensor for ML model
        """
        points, _ = self.process_image(image)
        
        # Convert to tensor (normalized coordinates)
        features = []
        h, w = image.shape[:2]
        
        for x, y, conf in points:
            # Normalize coordinates
            if conf > 0.1:
                features.extend([x / w, y / h, conf])
            else:
                features.extend([0.0, 0.0, 0.0])
        
        return torch.tensor(features, dtype=torch.float32)


class PoseDataset(Dataset):
    """Dataset that uses OpenPose for feature extraction"""
    
    def __init__(self, image_paths, labels, openpose_processor=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.openpose = openpose_processor
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply OpenPose
        if self.openpose:
            features = self.openpose.extract_features(image)
        else:
            # Fallback: use image directly
            if self.transform:
                image = self.transform(image)
            features = image
        
        return features, label