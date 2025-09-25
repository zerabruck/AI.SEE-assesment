#!/usr/bin/env python3
"""
Environment validation script for AI.SEE assessment.
This script verifies that all required packages are installed and working correctly.
"""

import sys
import importlib
from pathlib import Path

def validate_environment():
    """Validate that all required packages are installed and importable."""
    
    required_packages = [
        'torch',
        'torchvision', 
        'onnx',
        'onnxruntime',
        'ultralytics',
        'cv2',
        'numpy',
        'PIL',
        'matplotlib',
        'seaborn',
        'tqdm'
    ]
    
    print("🔍 Validating AI.SEE Assessment Environment...")
    print("=" * 50)
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"✅ OpenCV: {cv2.__version__}")
            elif package == 'PIL':
                from PIL import Image
                print(f"✅ Pillow: {Image.__version__}")
            else:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'Unknown')
                print(f"✅ {package}: {version}")
        except ImportError as e:
            print(f"❌ {package}: FAILED - {e}")
            failed_imports.append(package)
    
    print("=" * 50)
    
    if failed_imports:
        print(f"❌ Environment validation FAILED!")
        print(f"Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("✅ Environment validation PASSED!")
        print("All required packages are installed and working correctly.")
        return True

def check_pytorch_cuda():
    """Check if PyTorch can access CUDA (optional for this assessment)."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("💻 CUDA not available - using CPU (this is fine for the assessment)")
    except Exception as e:
        print(f"⚠️  CUDA check failed: {e}")

def check_ultralytics():
    """Check Ultralytics YOLO functionality."""
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO import successful")
        return True
    except Exception as e:
        print(f"❌ Ultralytics YOLO import failed: {e}")
        return False

if __name__ == "__main__":
    print("AI.SEE Assessment - Environment Validation")
    print("=" * 50)
    
    # Validate basic imports
    success = validate_environment()
    
    print("\n🔧 Additional Checks:")
    print("-" * 30)
    
    # Check CUDA availability
    check_pytorch_cuda()
    
    # Check Ultralytics
    ultralytics_ok = check_ultralytics()
    
    print("\n" + "=" * 50)
    if success and ultralytics_ok:
        print("🎉 Environment is ready for AI.SEE assessment!")
        sys.exit(0)
    else:
        print("⚠️  Environment has issues that need to be resolved.")
        sys.exit(1)
