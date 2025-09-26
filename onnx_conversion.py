#!/usr/bin/env python3
"""
ONNX Model Conversion Script
AI.SEE Assessment - Step 3: PyTorch to ONNX Conversion

This script converts the YOLO PyTorch model to ONNX format using Ultralytics export functionality.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import onnx
import onnxruntime as ort
from ultralytics import YOLO
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/onnx_conversion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class YOLOONNXConverter:
    """YOLO ONNX conversion class using Ultralytics export functionality."""
    
    def __init__(self, model_path: str):
        """
        Initialize the ONNX converter.
        
        Args:
            model_path: Path to the PyTorch model file
        """
        self.model_path = model_path
        self.model = None
        self.onnx_path = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the YOLO model."""
        try:
            logger.info(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def convert_to_onnx(self, output_dir: str = "output", 
                       imgsz: int = 640, 
                       half: bool = False,
                       dynamic: bool = True,
                       simplify: bool = True,
                       opset: int = 11,
                       int8: bool = False,
                       optimize: bool = False) -> str:
        """
        Convert PyTorch model to ONNX format using Ultralytics export.
        
        Args:
            output_dir: Directory to save the ONNX model
            imgsz: Input image size
            half: Use FP16 quantization
            dynamic: Enable dynamic input size
            simplify: Simplify the ONNX model
            opset: ONNX opset version
            int8: Enable INT8 quantization
            optimize: Apply optimizations for mobile/constrained environments
            
        Returns:
            Path to the converted ONNX model
        """
        try:
            logger.info("Starting ONNX conversion...")
            start_time = time.time()
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Export to ONNX using Ultralytics
            logger.info(f"Exporting model to ONNX format...")
            logger.info(f"Parameters: imgsz={imgsz}, half={half}, dynamic={dynamic}, simplify={simplify}, opset={opset}, int8={int8}, optimize={optimize}")
            
            # Use Ultralytics export method with all available options
            exported_path = self.model.export(
                format="onnx",
                imgsz=imgsz,
                half=half,
                dynamic=dynamic,
                simplify=simplify,
                opset=opset,
                int8=int8,
                optimize=optimize
            )
            
            conversion_time = time.time() - start_time
            logger.info(f"ONNX conversion completed in {conversion_time:.4f} seconds")
            logger.info(f"ONNX model saved to: {exported_path}")
            
            self.onnx_path = exported_path
            return exported_path
            
        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
            raise
    
    def validate_onnx_model(self, onnx_path: str) -> Dict[str, Any]:
        """
        Validate the converted ONNX model.
        
        Args:
            onnx_path: Path to the ONNX model
            
        Returns:
            Dictionary containing validation results
        """
        try:
            logger.info(f"Validating ONNX model: {onnx_path}")
            
            # Load ONNX model for validation
            onnx_model = onnx.load(onnx_path)
            
            # Check model validity
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model validation passed")
            
            # Get model information
            model_info = {
                'model_path': onnx_path,
                'model_size_mb': os.path.getsize(onnx_path) / (1024 * 1024),
                'input_shape': None,
                'output_shape': None,
                'opset_version': onnx_model.opset_import[0].version if onnx_model.opset_import else None
            }
            
            # Extract input/output shapes
            for input_tensor in onnx_model.graph.input:
                if input_tensor.name == 'images':  # YOLO input name
                    shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                            for dim in input_tensor.type.tensor_type.shape.dim]
                    model_info['input_shape'] = shape
                    break
            
            for output_tensor in onnx_model.graph.output:
                shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                        for dim in output_tensor.type.tensor_type.shape.dim]
                model_info['output_shape'] = shape
                break
            
            logger.info(f"Model size: {model_info['model_size_mb']:.2f} MB")
            logger.info(f"Input shape: {model_info['input_shape']}")
            logger.info(f"Output shape: {model_info['output_shape']}")
            logger.info(f"Opset version: {model_info['opset_version']}")
            
            return model_info
            
        except Exception as e:
            logger.error(f"ONNX model validation failed: {e}")
            raise
    
    def test_onnx_inference(self, onnx_path: str, image_path: str) -> Dict[str, Any]:
        """
        Test basic inference with the ONNX model.
        
        Args:
            onnx_path: Path to the ONNX model
            image_path: Path to test image
            
        Returns:
            Dictionary containing test results
        """
        try:
            logger.info(f"Testing ONNX inference with: {image_path}")
            
            # Create ONNX Runtime session
            session = ort.InferenceSession(onnx_path)
            
            # Get input/output details
            input_details = session.get_inputs()[0]
            output_details = session.get_outputs()
            
            logger.info(f"ONNX Runtime input: {input_details.name}, shape: {input_details.shape}")
            for i, output in enumerate(output_details):
                logger.info(f"ONNX Runtime output {i}: {output.name}, shape: {output.shape}")
            
            # Test inference
            start_time = time.time()
            results = self.model(image_path)  # Use original model for comparison
            inference_time = time.time() - start_time
            
            test_results = {
                'onnx_path': onnx_path,
                'test_image': image_path,
                'inference_time': inference_time,
                'input_name': input_details.name,
                'input_shape': input_details.shape,
                'output_names': [output.name for output in output_details],
                'output_shapes': [output.shape for output in output_details],
                'test_successful': True
            }
            
            logger.info("ONNX model test inference successful")
            return test_results
            
        except Exception as e:
            logger.error(f"ONNX inference test failed: {e}")
            return {
                'onnx_path': onnx_path,
                'test_image': image_path,
                'test_successful': False,
                'error': str(e)
            }
    
    def save_conversion_report(self, conversion_info: Dict[str, Any], 
                             validation_info: Dict[str, Any],
                             test_info: Dict[str, Any],
                             output_path: str) -> None:
        """
        Save conversion report to file.
        
        Args:
            conversion_info: Conversion process information
            validation_info: Model validation information
            test_info: Test inference information
            output_path: Path to save the report
        """
        try:
            with open(output_path, 'w') as f:
                f.write("YOLO PyTorch to ONNX Conversion Report\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("Conversion Information:\n")
                f.write("-" * 25 + "\n")
                f.write(f"Source Model: {self.model_path}\n")
                f.write(f"ONNX Model: {conversion_info.get('onnx_path', 'N/A')}\n")
                f.write(f"Conversion Time: {conversion_info.get('conversion_time', 'N/A')} seconds\n")
                f.write(f"Model Size: {validation_info.get('model_size_mb', 'N/A'):.2f} MB\n")
                f.write(f"Opset Version: {validation_info.get('opset_version', 'N/A')}\n\n")
                
                f.write("Model Architecture:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Input Shape: {validation_info.get('input_shape', 'N/A')}\n")
                f.write(f"Output Shape: {validation_info.get('output_shape', 'N/A')}\n\n")
                
                f.write("ONNX Runtime Information:\n")
                f.write("-" * 25 + "\n")
                f.write(f"Input Name: {test_info.get('input_name', 'N/A')}\n")
                f.write(f"Input Shape: {test_info.get('input_shape', 'N/A')}\n")
                f.write(f"Output Names: {test_info.get('output_names', 'N/A')}\n")
                f.write(f"Output Shapes: {test_info.get('output_shapes', 'N/A')}\n")
                f.write(f"Test Successful: {test_info.get('test_successful', 'N/A')}\n")
                
                if not test_info.get('test_successful', True):
                    f.write(f"Error: {test_info.get('error', 'N/A')}\n")
            
            logger.info(f"Conversion report saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save conversion report: {e}")
            raise


def main():
    """Main function to run ONNX conversion."""
    # Configuration
    model_path = "input/yolo11n.pt"
    image_path = "input/image-2.png"
    output_dir = "output"
    
    # Output paths
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    report_file = output_path / "onnx_conversion_report.txt"
    
    try:
        logger.info("Starting YOLO PyTorch to ONNX conversion...")
        
        # Initialize converter
        converter = YOLOONNXConverter(model_path)
        
        # Convert to ONNX
        onnx_path = converter.convert_to_onnx(
            output_dir=output_dir,
            imgsz=640,
            half=False,
            dynamic=True,
            simplify=True,
            opset=11
        )
        
        # Validate ONNX model
        validation_info = converter.validate_onnx_model(onnx_path)
        
        # Test ONNX inference
        test_info = converter.test_onnx_inference(onnx_path, image_path)
        
        # Save conversion report
        conversion_info = {
            'onnx_path': onnx_path,
            'conversion_time': 'N/A'  # Will be updated from logs
        }
        
        converter.save_conversion_report(
            conversion_info, 
            validation_info, 
            test_info, 
            report_file
        )
        
        logger.info("ONNX conversion completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("ONNX CONVERSION SUMMARY")
        print("="*50)
        print(f"Source Model: {model_path}")
        print(f"ONNX Model: {onnx_path}")
        print(f"Model Size: {validation_info['model_size_mb']:.2f} MB")
        print(f"Input Shape: {validation_info['input_shape']}")
        print(f"Output Shape: {validation_info['output_shape']}")
        print(f"Opset Version: {validation_info['opset_version']}")
        print(f"Test Successful: {test_info['test_successful']}")
        print(f"Report saved to: {report_file}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"ONNX conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
