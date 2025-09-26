#!/usr/bin/env python3
"""
PyTorch YOLO Model Inference Script
AI.SEE Assessment - Step 2: PyTorch Model Inference

This script loads the YOLO model in PyTorch format and runs inference on the provided image.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/pytorch_inference.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class YOLOPyTorchInference:
    """YOLO PyTorch inference class for object detection."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the YOLO PyTorch inference class.
        
        Args:
            model_path: Path to the PyTorch model file
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the YOLO model and extract metadata."""
        try:
            logger.info(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Get model information
            model_info = self.model.info()
            logger.info(f"Model loaded successfully")
            logger.info(f"Model type: {type(self.model.model)}")
            
            # Get class names
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                logger.info(f"Number of classes: {len(self.class_names)}")
                logger.info(f"Class names: {list(self.class_names.values())}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess the input image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Load image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            logger.info(f"Image loaded: {image_rgb.shape}")
            return image_rgb
            
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            raise
    
    def run_inference(self, image_path: str) -> Dict[str, Any]:
        """
        Run inference on the input image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing detection results
        """
        try:
            logger.info(f"Running inference on: {image_path}")
            start_time = time.time()
            
            # Run inference using Ultralytics YOLO
            results = self.model(image_path, conf=self.confidence_threshold)
            
            inference_time = time.time() - start_time
            logger.info(f"Inference completed in {inference_time:.4f} seconds")
            
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
            logger.error(f"Inference failed: {e}")
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
            image = self.preprocess_image(image_path)
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image)
            ax.set_title(f'YOLO PyTorch Detections (n={detections["num_detections"]})')
            ax.axis('off')
            
            # Draw bounding boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(detections['class_names'])))
            
            for i, (box, conf, class_name) in enumerate(zip(
                detections['boxes'], 
                detections['confidences'], 
                detections['class_names']
            )):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                # Create rectangle
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2, edgecolor=colors[i % len(colors)],
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                label = f'{class_name}: {conf:.2f}'
                ax.text(x1, y1-5, label, fontsize=10, color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i % len(colors)], alpha=0.7))
            
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
                f.write("YOLO PyTorch Inference Results\n")
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
    """Main function to run PyTorch inference."""
    # Configuration
    model_path = "input/yolo11n.pt"
    image_path = "input/image-2.png"
    confidence_threshold = 0.5
    
    # Output paths
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "pytorch_results.txt"
    visualization_file = output_dir / "pytorch_detections.png"
    
    try:
        logger.info("Starting PyTorch YOLO inference...")
        
        # Initialize inference class
        yolo_inference = YOLOPyTorchInference(model_path, confidence_threshold)
        
        # Run inference
        detections = yolo_inference.run_inference(image_path)
        
        # Save results
        yolo_inference.save_results(detections, results_file)
        
        # Create visualization
        yolo_inference.visualize_results(image_path, detections, visualization_file)
        
        logger.info("PyTorch inference completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("PYTORCH INFERENCE SUMMARY")
        print("="*50)
        print(f"Model: {model_path}")
        print(f"Image: {image_path}")
        print(f"Device: {yolo_inference.device}")
        print(f"Inference Time: {detections['inference_time']:.4f} seconds")
        print(f"Detections Found: {detections['num_detections']}")
        print(f"Results saved to: {results_file}")
        print(f"Visualization saved to: {visualization_file}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"PyTorch inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
