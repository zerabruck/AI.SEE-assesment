#!/usr/bin/env python3
"""
ONNX Model Inference Script
AI.SEE Assessment - Step 4: ONNX Model Inference

This script loads the converted ONNX model and runs inference on the provided image.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/onnx_inference.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class YOLOONNXInference:
    """YOLO ONNX inference class for object detection using Ultralytics."""
    
    def __init__(self, onnx_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the YOLO ONNX inference class.
        
        Args:
            onnx_path: Path to the ONNX model file
            confidence_threshold: Minimum confidence for detections
        """
        self.onnx_path = onnx_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the ONNX model using Ultralytics YOLO."""
        try:
            logger.info(f"Loading ONNX model from: {self.onnx_path}")
            
            # Load ONNX model using Ultralytics YOLO
            self.model = YOLO(self.onnx_path)
            
            # Get class names from the model
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                logger.info(f"Number of classes: {len(self.class_names)}")
                logger.info(f"Class names: {list(self.class_names.values())}")
            else:
                # Fallback to COCO class names if not available
                self.class_names = {
                    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
                    9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
                    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
                    33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
                    48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
                    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
                    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
                }
                logger.info(f"Using fallback COCO class names: {len(self.class_names)} classes")
            
            logger.info("ONNX model loaded successfully using Ultralytics")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    
    def run_inference(self, image_path: str) -> Dict[str, Any]:
        """
        Run inference on the input image using ONNX model via Ultralytics.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing detection results
        """
        try:
            logger.info(f"Running ONNX inference on: {image_path}")
            start_time = time.time()
            
            # Run inference using Ultralytics YOLO
            results = self.model(image_path, conf=self.confidence_threshold)
            
            inference_time = time.time() - start_time
            logger.info(f"ONNX inference completed in {inference_time:.4f} seconds")
            
            # Extract results
            result = results[0]  # Get first (and only) result
            
            # Get detection data
            boxes = result.boxes
            if boxes is not None:
                detections = {
                    'boxes': boxes.xyxy.cpu().numpy(),  # Bounding boxes
                    'confidences': boxes.conf.cpu().numpy(),  # Confidence scores
                    'class_ids': boxes.cls.cpu().numpy().astype(int),  # Class IDs
                    'class_names': [self.class_names[int(cls_id)] for cls_id in boxes.cls.cpu().numpy()],
                    'inference_time': inference_time,
                    'num_detections': len(boxes)
                }
            else:
                detections = {
                    'boxes': np.array([]),
                    'confidences': np.array([]),
                    'class_ids': np.array([]),
                    'class_names': [],
                    'inference_time': inference_time,
                    'num_detections': 0
                }
            
            logger.info(f"Found {detections['num_detections']} detections")
            return detections
            
        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            raise
    
    def visualize_results(self, image_path: str, detections: Dict[str, Any], 
                         output_path: str) -> None:
        """
        Create visualization of detection results.
        
        Args:
            image_path: Path to the original image
            detections: Detection results dictionary
            output_path: Path to save the visualization
        """
        try:
            # Load original image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image)
            ax.set_title(f'YOLO ONNX Detections (n={detections["num_detections"]})')
            ax.axis('off')
            
            # Draw bounding boxes with better color scheme
            # Use distinct colors for better visibility
            colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', 
                     '#FFA500', '#800080', '#FFC0CB', '#A52A2A', '#808080', '#000000']
            
            for i, (box, conf, class_name) in enumerate(zip(
                detections['boxes'], 
                detections['confidences'], 
                detections['class_names']
            )):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                # Get color for this detection
                color = colors[i % len(colors)]
                
                # Create rectangle with thicker border
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=3, edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label with better contrast
                label = f'{class_name}: {conf:.2f}'
                ax.text(x1, y1-8, label, fontsize=11, color='white', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.8, edgecolor='black', linewidth=1))
            
            # Save visualization
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
            raise
    
    def save_results(self, detections: Dict[str, Any], output_path: str) -> None:
        """
        Save detection results to a text file.
        
        Args:
            detections: Detection results dictionary
            output_path: Path to save the results
        """
        try:
            with open(output_path, 'w') as f:
                f.write("YOLO ONNX Inference Results\n")
                f.write("=" * 40 + "\n\n")
                
                f.write(f"Inference Time: {detections['inference_time']:.4f} seconds\n")
                f.write(f"Number of Detections: {detections['num_detections']}\n")
                f.write(f"Confidence Threshold: {self.confidence_threshold}\n\n")
                
                if detections['num_detections'] > 0:
                    f.write("Detections:\n")
                    f.write("-" * 20 + "\n")
                    
                    for i, (box, conf, class_name) in enumerate(zip(
                        detections['boxes'],
                        detections['confidences'],
                        detections['class_names']
                    )):
                        x1, y1, x2, y2 = box
                        f.write(f"Detection {i+1}:\n")
                        f.write(f"  Class: {class_name}\n")
                        f.write(f"  Confidence: {conf:.4f}\n")
                        f.write(f"  Bounding Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]\n")
                        f.write(f"  Width: {x2-x1:.1f}, Height: {y2-y1:.1f}\n\n")
                else:
                    f.write("No detections found.\n")
            
            logger.info(f"Results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise


def main():
    """Main function to run ONNX inference."""
    # Configuration
    onnx_path = "input/yolo11n.onnx"
    image_path = "input/image-2.png"
    confidence_threshold = 0.5
    
    # Output paths
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "onnx_results.txt"
    visualization_file = output_dir / "onnx_detections.png"
    
    try:
        logger.info("Starting ONNX YOLO inference...")
        
        # Initialize inference class
        yolo_inference = YOLOONNXInference(onnx_path, confidence_threshold)
        
        # Run inference
        detections = yolo_inference.run_inference(image_path)
        
        # Save results
        yolo_inference.save_results(detections, results_file)
        
        # Create visualization
        yolo_inference.visualize_results(image_path, detections, visualization_file)
        
        logger.info("ONNX inference completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("ONNX INFERENCE SUMMARY")
        print("="*50)
        print(f"Model: {onnx_path}")
        print(f"Image: {image_path}")
        print(f"Inference Time: {detections['inference_time']:.4f} seconds")
        print(f"Detections Found: {detections['num_detections']}")
        print(f"Results saved to: {results_file}")
        print(f"Visualization saved to: {visualization_file}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"ONNX inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
