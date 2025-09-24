#!/usr/bin/env python3
"""
ONNX Model Conversion Script for AI.SEE Assessment
This script converts a PyTorch YOLO model to ONNX format.
"""

import os
import time
import torch
import onnx
import onnxruntime as ort
from ultralytics import YOLO
from pathlib import Path
import numpy as np


class ONNXConverter:
    """Class to handle PyTorch to ONNX model conversion."""
    
    def __init__(self, pytorch_model_path, output_dir="output"):
        """
        Initialize the ONNX converter.
        
        Args:
            pytorch_model_path (str): Path to the PyTorch model file
            output_dir (str): Directory to save outputs
        """
        self.pytorch_model_path = pytorch_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.onnx_model_path = self.output_dir / "yolo11n.onnx"
        self.model = None
        self.conversion_time = 0
        
    def load_pytorch_model(self):
        """Load the PyTorch YOLO model."""
        print("Loading PyTorch YOLO model for conversion...")
        print(f"Model path: {self.pytorch_model_path}")
        
        try:
            self.model = YOLO(self.pytorch_model_path)
            print("✅ PyTorch model loaded successfully!")
            
            # Display model information
            print(f"Model type: {type(self.model.model)}")
            print(f"Model device: {self.model.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading PyTorch model: {e}")
            return False
    
    def convert_to_onnx(self):
        """Convert the PyTorch model to ONNX format."""
        print("Converting PyTorch model to ONNX...")
        print(f"Output path: {self.onnx_model_path}")
        
        try:
            start_time = time.time()
            
            # Use Ultralytics export functionality with proper parameters
            # The export method returns the path to the exported model
            exported_path = self.model.export(
                format='onnx',
                imgsz=640,  # Input image size
                optimize=True,  # Optimize the model
                half=False,  # Use FP32 precision
                dynamic=False,  # Use static input shape
                simplify=True,  # Simplify the model
                opset=11,  # ONNX opset version
                workspace=4,  # Workspace size in GB
                nms=True,  # Include NMS in the model
                batch=1  # Batch size
            )
            
            self.conversion_time = time.time() - start_time
            
            # Move the exported file to our desired location
            if os.path.exists(exported_path):
                if exported_path != str(self.onnx_model_path):
                    import shutil
                    shutil.move(exported_path, str(self.onnx_model_path))
                print(f"✅ ONNX model exported to: {self.onnx_model_path}")
            else:
                print(f"❌ Exported file not found at: {exported_path}")
                return False
            
            print(f"✅ ONNX conversion completed in {self.conversion_time:.4f} seconds")
            return True
            
        except Exception as e:
            print(f"❌ Error during ONNX conversion: {e}")
            return False
    
    def validate_onnx_model(self):
        """Validate the converted ONNX model."""
        print("Validating ONNX model...")
        
        try:
            # Check if the ONNX file exists
            if not self.onnx_model_path.exists():
                print(f"❌ ONNX model file not found: {self.onnx_model_path}")
                return False
            
            # Load and validate the ONNX model
            onnx_model = onnx.load(str(self.onnx_model_path))
            onnx.checker.check_model(onnx_model)
            
            print("✅ ONNX model validation passed!")
            
            # Display model information
            print(f"ONNX model file size: {self.onnx_model_path.stat().st_size / (1024*1024):.2f} MB")
            
            # Get model input/output information
            print("\nModel Input/Output Information:")
            print("-" * 40)
            
            for input_info in onnx_model.graph.input:
                print(f"Input: {input_info.name}")
                if input_info.type.tensor_type.shape:
                    shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                            for dim in input_info.type.tensor_type.shape.dim]
                    print(f"  Shape: {shape}")
                print(f"  Type: {input_info.type.tensor_type.elem_type}")
            
            for output_info in onnx_model.graph.output:
                print(f"Output: {output_info.name}")
                if output_info.type.tensor_type.shape:
                    shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                            for dim in output_info.type.tensor_type.shape.dim]
                    print(f"  Shape: {shape}")
                print(f"  Type: {output_info.type.tensor_type.elem_type}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error validating ONNX model: {e}")
            return False
    
    def test_onnx_inference(self):
        """Test basic inference with the ONNX model."""
        print("Testing ONNX model inference...")
        
        try:
            # Create ONNX Runtime session
            session = ort.InferenceSession(str(self.onnx_model_path))
            
            # Get input details
            input_details = session.get_inputs()[0]
            input_name = input_details.name
            input_shape = input_details.shape
            
            print(f"Input name: {input_name}")
            print(f"Input shape: {input_shape}")
            
            # Create dummy input for testing
            # Handle dynamic batch size
            if input_shape[0] == -1 or input_shape[0] == 'batch_size':
                input_shape[0] = 1
            
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Run inference
            start_time = time.time()
            outputs = session.run(None, {input_name: dummy_input})
            inference_time = time.time() - start_time
            
            print(f"✅ ONNX inference test successful!")
            print(f"Test inference time: {inference_time:.4f} seconds")
            print(f"Number of outputs: {len(outputs)}")
            
            # Display output shapes
            for i, output in enumerate(outputs):
                print(f"Output {i} shape: {output.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error testing ONNX inference: {e}")
            return False
    
    def save_conversion_report(self):
        """Save a detailed conversion report."""
        try:
            report_file = self.output_dir / "onnx_conversion_report.txt"
            
            with open(report_file, 'w') as f:
                f.write("ONNX CONVERSION REPORT\n")
                f.write("="*50 + "\n\n")
                f.write(f"Source PyTorch model: {self.pytorch_model_path}\n")
                f.write(f"Output ONNX model: {self.onnx_model_path}\n")
                f.write(f"Conversion time: {self.conversion_time:.4f} seconds\n")
                f.write(f"ONNX model size: {self.onnx_model_path.stat().st_size / (1024*1024):.2f} MB\n\n")
                
                # Get model information
                if self.onnx_model_path.exists():
                    onnx_model = onnx.load(str(self.onnx_model_path))
                    
                    f.write("MODEL STRUCTURE:\n")
                    f.write("-" * 30 + "\n")
                    
                    f.write("Inputs:\n")
                    for input_info in onnx_model.graph.input:
                        f.write(f"  - {input_info.name}\n")
                        if input_info.type.tensor_type.shape:
                            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                                   for dim in input_info.type.tensor_type.shape.dim]
                            f.write(f"    Shape: {shape}\n")
                        f.write(f"    Type: {input_info.type.tensor_type.elem_type}\n")
                    
                    f.write("\nOutputs:\n")
                    for output_info in onnx_model.graph.output:
                        f.write(f"  - {output_info.name}\n")
                        if output_info.type.tensor_type.shape:
                            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                                   for dim in output_info.type.tensor_type.shape.dim]
                            f.write(f"    Shape: {shape}\n")
                        f.write(f"    Type: {output_info.type.tensor_type.elem_type}\n")
            
            print(f"✅ Conversion report saved to: {report_file}")
            
        except Exception as e:
            print(f"❌ Error saving conversion report: {e}")
    
    def run_conversion(self):
        """Run the complete conversion pipeline."""
        print("Starting ONNX Conversion Pipeline")
        print("="*50)
        
        # Load PyTorch model
        if not self.load_pytorch_model():
            return False
        
        # Convert to ONNX
        if not self.convert_to_onnx():
            return False
        
        # Validate ONNX model
        if not self.validate_onnx_model():
            return False
        
        # Test ONNX inference
        if not self.test_onnx_inference():
            return False
        
        # Save conversion report
        self.save_conversion_report()
        
        print("\n✅ ONNX conversion pipeline completed successfully!")
        print(f"ONNX model saved to: {self.onnx_model_path}")
        return True


def main():
    """Main function to run ONNX conversion."""
    # Define paths
    pytorch_model_path = "assets/yolo11n.pt"
    output_dir = "output"
    
    # Check if PyTorch model exists
    if not os.path.exists(pytorch_model_path):
        print(f"❌ PyTorch model file not found: {pytorch_model_path}")
        return
    
    # Create converter instance
    converter = ONNXConverter(pytorch_model_path, output_dir)
    
    # Run conversion
    success = converter.run_conversion()
    
    if success:
        print("\n🎉 ONNX conversion completed successfully!")
    else:
        print("\n❌ ONNX conversion failed!")


if __name__ == "__main__":
    main()
