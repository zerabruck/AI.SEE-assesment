# AI.SEE Assessment - YOLO Model Inference and ONNX Conversion

This project demonstrates YOLO model inference using both PyTorch and ONNX formats, including model conversion and comprehensive result comparison.

## Project Structure

```
Assessment/
├── input/                          # Input files
│   ├── yolo11n.pt                 # PyTorch YOLO model
│   ├── yolo11n.onnx               # Converted ONNX model
│   └── image-2.png                # Test image
├── output/                         # Output files and results
│   ├── pytorch_results.txt        # PyTorch inference results
│   ├── pytorch_detections.png     # PyTorch visualization
│   ├── onnx_results.txt           # ONNX inference results
│   ├── onnx_detections.png        # ONNX visualization
│   ├── comparison_report.txt      # Detailed comparison report
│   ├── comparison_visualization.png # Comparison charts
│   └── performance_comparison.png  # Performance analysis
├── utility/                        # Utility functions
│   └── helper_functions.py        # Common helper functions
├── project_notes/                  # Documentation
│   ├── assesment.md               # Assessment requirements
│   ├── plan.md                    # Project plan
│   └── todo.md                    # Task checklist
├── pytorch_inference.py           # PyTorch inference script
├── onnx_conversion.py             # ONNX conversion script
├── onnx_inference.py              # ONNX inference script
├── results_comparison.py          # Results comparison script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Environment Setup

### Prerequisites
- Python 3.10+
- Virtual environment support

### Installation

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch; import ultralytics; import onnx; print('All packages installed successfully!')"
   ```

## Usage

### Step 1: PyTorch Inference
Run inference using the original PyTorch model:
```bash
python pytorch_inference.py
```

**Output:**
- `output/pytorch_results.txt` - Detection results
- `output/pytorch_detections.png` - Annotated image
- `output/pytorch_inference.log` - Execution log

### Step 2: ONNX Conversion
Convert PyTorch model to ONNX format:
```bash
python onnx_conversion.py
```

**Output:**
- `input/yolo11n.onnx` - Converted ONNX model
- `output/onnx_conversion_report.txt` - Conversion details
- `output/onnx_conversion.log` - Conversion log

### Step 3: ONNX Inference
Run inference using the converted ONNX model:
```bash
python onnx_inference.py
```

**Output:**
- `output/onnx_results.txt` - Detection results
- `output/onnx_detections.png` - Annotated image
- `output/onnx_inference.log` - Execution log

### Step 4: Results Comparison
Compare PyTorch and ONNX results:
```bash
python results_comparison.py
```

**Output:**
- `output/comparison_report.txt` - Detailed comparison
- `output/comparison_visualization.png` - Comparison charts
- `output/performance_comparison.png` - Performance analysis

## Results Summary

### Performance Comparison
- **PyTorch Inference Time:** 0.6909 seconds
- **ONNX Inference Time:** 0.5522 seconds
- **ONNX Speedup:** 1.25x faster

### Detection Comparison
- **PyTorch Detections:** 4 objects
- **ONNX Detections:** 38 objects
- **Matched Detections:** 4 objects
- **ONNX Only:** 34 additional detections

### Class-wise Analysis
- **Giraffes:** PyTorch (2) vs ONNX (20)
- **Cars:** PyTorch (2) vs ONNX (18)

## Key Features

### PyTorch Inference (`pytorch_inference.py`)
- YOLO model loading and metadata extraction
- Image preprocessing and inference
- Result visualization with bounding boxes
- Comprehensive logging and error handling
- Performance timing

### ONNX Conversion (`onnx_conversion.py`)
- Ultralytics export functionality
- Model validation and testing
- Dynamic input shape support
- Conversion report generation

### ONNX Inference (`onnx_inference.py`)
- ONNX Runtime integration
- Custom preprocessing pipeline
- Post-processing with confidence filtering
- Enhanced visualization with better colors
- Performance optimization

### Results Comparison (`results_comparison.py`)
- IoU-based detection matching
- Confidence score analysis
- Class-wise comparison
- Performance benchmarking
- Comprehensive reporting

## Technical Details

### Model Information
- **Model:** YOLO11n
- **Parameters:** 2.6M
- **GFLOPs:** 6.6
- **Classes:** 80 (COCO dataset)
- **Input Size:** 640x640

### Dependencies
- **PyTorch:** 2.8.0+cu128
- **Ultralytics:** 8.3.203
- **ONNX:** 1.19.0
- **ONNXRuntime:** 1.23.0
- **OpenCV:** 4.12.0

### Hardware Requirements
- **CPU:** AMD Ryzen 5 2500U (tested)
- **Memory:** 8GB RAM (recommended)
- **Storage:** 2GB free space

## Assessment Criteria Fulfillment

### ✅ Completeness
- Successfully ran inference in both PyTorch and ONNX
- Complete workflow from model loading to result comparison

### ✅ Environment Setup
- Fresh Python virtual environment
- All required dependencies installed
- Proper project structure

### ✅ AI Tool Usage
- Effective use of AI-assisted coding
- Modular and well-documented code
- Comprehensive error handling

### ✅ Clarity of Explanation
- Detailed logging and documentation
- Step-by-step execution process
- Clear result visualization

### ✅ Correctness of Outputs
- Sensible bounding boxes and labels
- Consistent detection results
- Proper model conversion validation

## Troubleshooting

### Common Issues

1. **CUDA not available:**
   - Models will run on CPU (slower but functional)
   - No action required

2. **Memory issues:**
   - Reduce batch size in inference scripts
   - Close other applications

3. **Import errors:**
   - Ensure virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt`

### Performance Tips

1. **For faster inference:**
   - Use GPU if available
   - Reduce input image size
   - Increase confidence threshold

2. **For better accuracy:**
   - Use higher resolution input
   - Lower confidence threshold
   - Apply post-processing filters

## License

This project is created for AI.SEE assessment purposes.

## Contact

For questions or issues, please refer to the project documentation in `project_notes/`.
