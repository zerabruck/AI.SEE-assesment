#!/usr/bin/env python3
"""
PyTorch YOLO Model Inference Script for AI.SEE Assessment
This script loads a YOLO model and runs inference on an image using PyTorch.
"""

import os
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path


class PyTorchYOLOInference:
    """Class to handle PyTorch YOLO model inference."""
    
    def __init__(self, model_path, image_path, output_dir="output"):
        """
        Initialize the PyTorch YOLO inference.
        
        Args:
            model_path (str): Path to the PyTorch model file
            image_path (str): Path to the input image
            output_dir (str): Directory to save outputs
        """
        self.model_path = model_path
        self.image_path = image_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load the model
        self.model = None
        self.results = None
        self.inference_time = 0
        
    def load_model(self):
        """Load the YOLO model and display model information."""
        print("Loading PyTorch YOLO model...")
        print(f"Model path: {self.model_path}")
        
        try:
            self.model = YOLO(self.model_path)
            print("✅ Model loaded successfully!")
            
            # Display model information
            print(f"Model type: {type(self.model.model)}")
            print(f"Model device: {self.model.device}")
            
            # Get model metadata
            if hasattr(self.model.model, 'names'):
                print(f"Number of classes: {len(self.model.model.names)}")
                print(f"Class names: {list(self.model.model.names.values())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def preprocess_image(self):
        """Load and preprocess the input image."""
        print(f"Loading image: {self.image_path}")
        
        try:
            # Load image using OpenCV
            image = cv2.imread(self.image_path)
            if image is None:
                raise ValueError(f"Could not load image from {self.image_path}")
            
            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            print(f"Image shape: {image.shape}")
            print(f"Image dtype: {image.dtype}")
            
            return image, image_rgb
            
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None, None
    
    def run_inference(self):
        """Run inference on the image."""
        print("Running PyTorch inference...")
        
        try:
            start_time = time.time()
            
            # Run inference using Ultralytics YOLO
            self.results = self.model(self.image_path)
            
            self.inference_time = time.time() - start_time
            
            print(f"✅ Inference completed in {self.inference_time:.4f} seconds")
            return True
            
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            return False
    
    def process_results(self):
        """Process and display the inference results."""
        if self.results is None:
            print("❌ No results to process")
            return None
        
        print("\n" + "="*50)
        print("PYTORCH INFERENCE RESULTS")
        print("="*50)
        
        # Process each result
        all_detections = []
        
        for i, result in enumerate(self.results):
            print(f"\nResult {i+1}:")
            
            # Get detection data
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                print(f"Number of detections: {len(boxes)}")
                
                if len(boxes) > 0:
                    print("\nDetections:")
                    print("-" * 80)
                    print(f"{'Class':<15} {'Confidence':<12} {'BBox (x1,y1,x2,y2)':<25} {'Area':<10}")
                    print("-" * 80)
                    
                    for j, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                        class_name = self.model.names[cls_id] if cls_id in self.model.names else f"Class_{cls_id}"
                        area = (box[2] - box[0]) * (box[3] - box[1])
                        
                        print(f"{class_name:<15} {conf:<12.4f} {str(box):<25} {area:<10.1f}")
                        
                        all_detections.append({
                            'class_name': class_name,
                            'confidence': float(conf),
                            'bbox': box.tolist(),
                            'area': float(area)
                        })
                else:
                    print("No objects detected")
            else:
                print("No detection results available")
        
        return all_detections
    
    def visualize_results(self, image_rgb, detections):
        """Create visualization of the detection results."""
        print("Creating visualization...")
        
        try:
            # Create a copy of the image for annotation
            annotated_image = image_rgb.copy()
            
            # Draw bounding boxes and labels
            for detection in detections:
                bbox = detection['bbox']
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                # Convert to integer coordinates
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Draw label background
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(annotated_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Save the annotated image
            output_path = self.output_dir / "pytorch_detections.png"
            cv2.imwrite(str(output_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            print(f"✅ Annotated image saved to: {output_path}")
            
            return annotated_image
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
            return None
    
    def save_results(self, detections):
        """Save detailed results to a text file."""
        try:
            results_file = self.output_dir / "pytorch_results.txt"
            
            with open(results_file, 'w') as f:
                f.write("PYTORCH YOLO INFERENCE RESULTS\n")
                f.write("="*50 + "\n\n")
                f.write(f"Model: {self.model_path}\n")
                f.write(f"Image: {self.image_path}\n")
                f.write(f"Inference time: {self.inference_time:.4f} seconds\n")
                f.write(f"Number of detections: {len(detections)}\n\n")
                
                if detections:
                    f.write("DETECTION DETAILS:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{'Class':<15} {'Confidence':<12} {'BBox (x1,y1,x2,y2)':<25} {'Area':<10}\n")
                    f.write("-" * 80 + "\n")
                    
                    for detection in detections:
                        f.write(f"{detection['class_name']:<15} {detection['confidence']:<12.4f} "
                               f"{str(detection['bbox']):<25} {detection['area']:<10.1f}\n")
                else:
                    f.write("No objects detected\n")
            
            print(f"✅ Results saved to: {results_file}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def run_complete_inference(self):
        """Run the complete inference pipeline."""
        print("Starting PyTorch YOLO Inference Pipeline")
        print("="*50)
        
        # Load model
        if not self.load_model():
            return False
        
        # Load and preprocess image
        image, image_rgb = self.preprocess_image()
        if image is None:
            return False
        
        # Run inference
        if not self.run_inference():
            return False
        
        # Process results
        detections = self.process_results()
        
        # Create visualization
        if detections:
            self.visualize_results(image_rgb, detections)
        
        # Save results
        self.save_results(detections)
        
        print("\n✅ PyTorch inference pipeline completed successfully!")
        return True


def main():
    """Main function to run PyTorch inference."""
    # Define paths
    model_path = "assets/yolo11n.pt"
    image_path = "assets/image-2.png"
    output_dir = "output"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    # Create inference instance
    inference = PyTorchYOLOInference(model_path, image_path, output_dir)
    
    # Run complete inference
    success = inference.run_complete_inference()
    
    if success:
        print("\n🎉 PyTorch inference completed successfully!")
    else:
        print("\n❌ PyTorch inference failed!")


if __name__ == "__main__":
    main()
