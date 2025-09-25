#!/usr/bin/env python3
"""
ONNX Model Inference using Ultralytics for AI.SEE Assessment
This script demonstrates ONNX model inference using the Ultralytics framework.
"""

import os
import time
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

class UltralyticsONNXInference:
    """ONNX inference pipeline using Ultralytics framework."""
    
    def __init__(self, model_path, image_path, output_dir="output"):
        """
        Initialize the Ultralytics ONNX inference pipeline.
        
        Args:
            model_path (str): Path to the ONNX model file (.onnx)
            image_path (str): Path to the input image
            output_dir (str): Output directory for results
        """
        self.model_path = model_path
        self.image_path = image_path
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / "results"
        self.logs_dir = self.output_dir / "logs"
        
        # Create output directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.results = None
        self.inference_time = 0
        
    def load_model(self):
        """Load the ONNX model using Ultralytics."""
        print(f"🔄 Loading ONNX model from: {self.model_path}")
        
        try:
            # Load ONNX model using Ultralytics
            self.model = YOLO(self.model_path)
            print(f"✅ ONNX model loaded successfully!")
            print(f"   Model type: {type(self.model.model)}")
            print(f"   Device: {self.model.device}")
            return True
        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}")
            return False
    
    def preprocess_image(self):
        """Load and preprocess the input image."""
        print(f"🔄 Loading image from: {self.image_path}")
        
        try:
            # Load image using OpenCV
            self.image = cv2.imread(self.image_path)
            if self.image is None:
                raise ValueError(f"Could not load image from {self.image_path}")
            
            # Convert BGR to RGB for display
            self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            
            print(f"✅ Image loaded successfully!")
            print(f"   Image shape: {self.image.shape}")
            print(f"   Image dtype: {self.image.dtype}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load image: {e}")
            return False
    
    def run_inference(self, conf_threshold=0.25, iou_threshold=0.45):
        """
        Run inference on the loaded image using Ultralytics ONNX model.
        
        Args:
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
        """
        print(f"🔄 Running ONNX inference...")
        print(f"   Confidence threshold: {conf_threshold}")
        print(f"   IoU threshold: {iou_threshold}")
        
        try:
            start_time = time.time()
            
            # Run inference using Ultralytics ONNX model
            self.results = self.model(
                self.image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            self.inference_time = time.time() - start_time
            
            print(f"✅ ONNX inference completed!")
            print(f"   Inference time: {self.inference_time:.4f} seconds")
            print(f"   Number of detections: {len(self.results[0].boxes) if self.results[0].boxes is not None else 0}")
            
            return True
            
        except Exception as e:
            print(f"❌ ONNX inference failed: {e}")
            return False
    
    def extract_detections(self):
        """Extract detection results from the model output."""
        if self.results is None:
            print("❌ No inference results available")
            return None
        
        detections = []
        result = self.results[0]  # Get first (and only) result
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            class_names = result.names
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                detection = {
                    'id': i,
                    'bbox': {
                        'x1': float(box[0]),
                        'y1': float(box[1]),
                        'x2': float(box[2]),
                        'y2': float(box[3]),
                        'width': float(box[2] - box[0]),
                        'height': float(box[3] - box[1])
                    },
                    'confidence': float(conf),
                    'class_id': int(cls_id),
                    'class_name': class_names[cls_id]
                }
                detections.append(detection)
        
        print(f"📊 Extracted {len(detections)} detections")
        return detections
    
    def visualize_results(self, save_path=None):
        """Create visualization of detection results."""
        if self.results is None:
            print("❌ No results to visualize")
            return None
        
        print("🔄 Creating ONNX visualization...")
        
        try:
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(self.image_rgb)
            ax.set_title(f'ONNX YOLO Detections (Inference Time: {self.inference_time:.4f}s)', 
                        fontsize=14, fontweight='bold')
            
            # Get detections
            detections = self.extract_detections()
            
            if detections:
                # Color map for different classes
                colors = plt.cm.Set3(np.linspace(0, 1, len(set(d['class_id'] for d in detections))))
                class_colors = {cls_id: colors[i] for i, cls_id in enumerate(set(d['class_id'] for d in detections))}
                
                for detection in detections:
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                    
                    # Create rectangle
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=2,
                        edgecolor=class_colors[detection['class_id']],
                        facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Add label
                    label = f"{detection['class_name']}: {detection['confidence']:.2f}"
                    ax.text(x1, y1-5, label, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=class_colors[detection['class_id']], alpha=0.7),
                           fontsize=10, fontweight='bold')
            
            ax.axis('off')
            plt.tight_layout()
            
            # Save visualization
            if save_path is None:
                save_path = self.results_dir / "onnx_results_ultralytics.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ ONNX visualization saved to: {save_path}")
            
            plt.show()
            return save_path
            
        except Exception as e:
            print(f"❌ ONNX visualization failed: {e}")
            return None
    
    def save_results(self):
        """Save detection results to JSON file."""
        detections = self.extract_detections()
        
        if detections is None:
            print("❌ No detections to save")
            return None
        
        # Prepare results data
        results_data = {
            'model_info': {
                'model_path': str(self.model_path),
                'model_type': 'ONNX YOLO (Ultralytics)',
                'framework': 'Ultralytics ONNX',
                'device': str(self.model.device),
                'inference_time': self.inference_time
            },
            'image_info': {
                'image_path': str(self.image_path),
                'image_shape': self.image.shape,
                'image_dtype': str(self.image.dtype)
            },
            'detection_summary': {
                'total_detections': len(detections),
                'confidence_threshold': 0.25,
                'timestamp': datetime.now().isoformat()
            },
            'detections': detections
        }
        
        # Save to JSON
        json_path = self.results_dir / "onnx_detections_ultralytics.json"
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"✅ ONNX results saved to: {json_path}")
        return json_path
    
    def log_inference_details(self):
        """Log detailed inference information."""
        log_path = self.logs_dir / "onnx_inference_ultralytics.log"
        
        with open(log_path, 'w') as f:
            f.write("ONNX YOLO Inference Log (Ultralytics)\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Image Path: {self.image_path}\n")
            f.write(f"Device: {self.model.device}\n")
            f.write(f"Inference Time: {self.inference_time:.4f} seconds\n\n")
            
            if self.results:
                result = self.results[0]
                f.write(f"Number of detections: {len(result.boxes) if result.boxes is not None else 0}\n")
                
                if result.boxes is not None and len(result.boxes) > 0:
                    f.write("\nDetection Details:\n")
                    f.write("-" * 30 + "\n")
                    
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    class_names = result.names
                    
                    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                        f.write(f"Detection {i+1}:\n")
                        f.write(f"  Class: {class_names[cls_id]} (ID: {cls_id})\n")
                        f.write(f"  Confidence: {conf:.4f}\n")
                        f.write(f"  Bounding Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]\n")
                        f.write(f"  Width: {box[2]-box[0]:.1f}, Height: {box[3]-box[1]:.1f}\n\n")
        
        print(f"✅ ONNX inference log saved to: {log_path}")
        return log_path
    
    def run_full_pipeline(self):
        """Run the complete ONNX inference pipeline."""
        print("🚀 Starting ONNX YOLO Inference Pipeline (Ultralytics)")
        print("=" * 60)
        
        # Step 1: Load ONNX model
        if not self.load_model():
            return False
        
        # Step 2: Preprocess image
        if not self.preprocess_image():
            return False
        
        # Step 3: Run inference
        if not self.run_inference():
            return False
        
        # Step 4: Extract and save results
        self.save_results()
        
        # Step 5: Create visualization
        self.visualize_results()
        
        # Step 6: Log details
        self.log_inference_details()
        
        print("\n✅ ONNX inference pipeline completed successfully!")
        return True

def main():
    """Main function to run ONNX inference."""
    # Define paths
    model_path = "output/models/yolo11n.onnx"
    image_path = "input/image-2.png"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ ONNX model file not found: {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    # Initialize and run inference
    inference = UltralyticsONNXInference(model_path, image_path)
    success = inference.run_full_pipeline()
    
    if success:
        print("\n🎉 ONNX inference completed successfully!")
        print("Check the 'output/results/' directory for visualization and results.")
    else:
        print("\n❌ ONNX inference failed!")

if __name__ == "__main__":
    main()
