#!/usr/bin/env python3
"""
ONNX Model Inference Script for AI.SEE Assessment
This script loads an ONNX YOLO model and runs inference on an image using Ultralytics.
"""

import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


class ONNXYOLOInference:
    """Class to handle ONNX YOLO model inference using Ultralytics."""
    
    def __init__(self, onnx_model_path, image_path, output_dir="output"):
        """
        Initialize the ONNX YOLO inference.
        
        Args:
            onnx_model_path (str): Path to the ONNX model file
            image_path (str): Path to the input image
            output_dir (str): Directory to save outputs
        """
        self.onnx_model_path = onnx_model_path
        self.image_path = image_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load the model
        self.model = None
        self.results = None
        self.inference_time = 0
        
    def load_model(self):
        """Load the ONNX YOLO model using Ultralytics."""
        print("Loading ONNX YOLO model using Ultralytics...")
        print(f"Model path: {self.onnx_model_path}")
        
        try:
            # Load ONNX model using Ultralytics
            self.model = YOLO(self.onnx_model_path)
            print("✅ ONNX model loaded successfully!")
            
            # Display model information
            print(f"Model type: {type(self.model.model)}")
            print(f"Model device: {self.model.device}")
            
            # Get model metadata
            if hasattr(self.model.model, 'names'):
                print(f"Number of classes: {len(self.model.model.names)}")
                print(f"Class names: {list(self.model.model.names.values())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading ONNX model: {e}")
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
        """Run inference on the image using ONNX model."""
        print("Running ONNX inference...")
        
        try:
            start_time = time.time()
            
            # Run inference using Ultralytics YOLO with ONNX model
            self.results = self.model(self.image_path)
            
            self.inference_time = time.time() - start_time
            
            print(f"✅ ONNX inference completed in {self.inference_time:.4f} seconds")
            return True
            
        except Exception as e:
            print(f"❌ Error during ONNX inference: {e}")
            return False
    
    def process_results(self):
        """Process and display the inference results."""
        if self.results is None:
            print("❌ No results to process")
            return None
        
        print("\n" + "="*50)
        print("ONNX INFERENCE RESULTS")
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
        print("Creating ONNX visualization...")
        
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
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for ONNX
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Draw label background
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (255, 0, 0), -1)  # Blue background
                
                # Draw label text
                cv2.putText(annotated_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # White text
            
            # Save the annotated image
            output_path = self.output_dir / "onnx_detections.png"
            cv2.imwrite(str(output_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            print(f"✅ ONNX annotated image saved to: {output_path}")
            
            return annotated_image
            
        except Exception as e:
            print(f"❌ Error creating ONNX visualization: {e}")
            return None
    
    def save_results(self, detections):
        """Save detailed results to a text file."""
        try:
            results_file = self.output_dir / "onnx_results.txt"
            
            with open(results_file, 'w') as f:
                f.write("ONNX YOLO INFERENCE RESULTS\n")
                f.write("="*50 + "\n\n")
                f.write(f"Model: {self.onnx_model_path}\n")
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
            
            print(f"✅ ONNX results saved to: {results_file}")
            
        except Exception as e:
            print(f"❌ Error saving ONNX results: {e}")
    
    def run_complete_inference(self):
        """Run the complete ONNX inference pipeline."""
        print("Starting ONNX YOLO Inference Pipeline")
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
        
        print("\n✅ ONNX inference pipeline completed successfully!")
        return True


def main():
    """Main function to run ONNX inference."""
    # Define paths
    onnx_model_path = "output/yolo11n.onnx"
    image_path = "assets/image-2.png"
    output_dir = "output"
    
    # Check if files exist
    if not os.path.exists(onnx_model_path):
        print(f"❌ ONNX model file not found: {onnx_model_path}")
        print("Please run the ONNX conversion script first.")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    # Create inference instance
    inference = ONNXYOLOInference(onnx_model_path, image_path, output_dir)
    
    # Run complete inference
    success = inference.run_complete_inference()
    
    if success:
        print("\n🎉 ONNX inference completed successfully!")
    else:
        print("\n❌ ONNX inference failed!")


if __name__ == "__main__":
    main()
