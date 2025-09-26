#!/usr/bin/env python3
"""
Helper Functions for YOLO Model Inference
AI.SEE Assessment - Utility Functions

This module contains utility functions used across the inference scripts.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import cv2


def setup_logging(log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create output directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, 
                       iou_threshold: float = 0.5) -> List[int]:
    """
    Apply Non-Maximum Suppression to filter overlapping detections.
    
    Args:
        boxes: Array of bounding boxes [N, 4] in format [x1, y1, x2, y2]
        scores: Array of confidence scores [N]
        iou_threshold: IoU threshold for suppression
        
    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    # Sort by confidence scores (descending)
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        # Pick the box with highest confidence
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current]
        remaining_boxes = boxes[indices[1:]]
        
        ious = []
        for box in remaining_boxes:
            ious.append(calculate_iou(current_box, box))
        
        # Keep boxes with IoU below threshold
        ious = np.array(ious)
        indices = indices[1:][ious <= iou_threshold]
    
    return keep


def resize_image(image: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, float, float]:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image as numpy array
        target_size: Target size for the longer side
        
    Returns:
        Tuple of (resized_image, scale_x, scale_y)
    """
    h, w = image.shape[:2]
    
    # Calculate scale factor
    scale = target_size / max(h, w)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    return resized, scale, scale


def pad_image(image: np.ndarray, target_size: int = 640) -> np.ndarray:
    """
    Pad image to target size with black pixels.
    
    Args:
        image: Input image as numpy array
        target_size: Target size for both dimensions
        
    Returns:
        Padded image
    """
    h, w = image.shape[:2]
    
    # Calculate padding
    pad_h = target_size - h
    pad_w = target_size - w
    
    # Pad with zeros (black)
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    
    return padded


def get_coco_class_names() -> List[str]:
    """
    Get COCO dataset class names.
    
    Returns:
        List of class names
    """
    return [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]


def validate_file_path(file_path: str, file_type: str = "file") -> bool:
    """
    Validate if file path exists and is accessible.
    
    Args:
        file_path: Path to the file
        file_type: Type of file ("file" or "directory")
        
    Returns:
        True if valid, False otherwise
    """
    path = Path(file_path)
    
    if file_type == "file":
        return path.exists() and path.is_file()
    elif file_type == "directory":
        return path.exists() and path.is_dir()
    else:
        return path.exists()


def create_output_directory(output_path: str) -> None:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_path: Path to the output directory
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)


def format_confidence(confidence: float, decimals: int = 4) -> str:
    """
    Format confidence score for display.
    
    Args:
        confidence: Confidence score
        decimals: Number of decimal places
        
    Returns:
        Formatted confidence string
    """
    return f"{confidence:.{decimals}f}"


def format_bbox(bbox: List[float], decimals: int = 1) -> str:
    """
    Format bounding box for display.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        decimals: Number of decimal places
        
    Returns:
        Formatted bounding box string
    """
    return f"[{bbox[0]:.{decimals}f}, {bbox[1]:.{decimals}f}, {bbox[2]:.{decimals}f}, {bbox[3]:.{decimals}f}]"


def calculate_bbox_area(bbox: List[float]) -> float:
    """
    Calculate area of bounding box.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Area of the bounding box
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def calculate_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Calculate center point of bounding box.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Tuple of (center_x, center_y)
    """
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return center_x, center_y


def filter_detections_by_confidence(detections: List[Dict[str, Any]], 
                                  confidence_threshold: float) -> List[Dict[str, Any]]:
    """
    Filter detections by confidence threshold.
    
    Args:
        detections: List of detection dictionaries
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        Filtered list of detections
    """
    return [det for det in detections if det.get('confidence', 0) >= confidence_threshold]


def sort_detections_by_confidence(detections: List[Dict[str, Any]], 
                                descending: bool = True) -> List[Dict[str, Any]]:
    """
    Sort detections by confidence score.
    
    Args:
        detections: List of detection dictionaries
        descending: Sort in descending order (highest first)
        
    Returns:
        Sorted list of detections
    """
    return sorted(detections, key=lambda x: x.get('confidence', 0), reverse=descending)
