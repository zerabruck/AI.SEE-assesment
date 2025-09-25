# AI.SEE Assessment - YOLO Model Inference

This project demonstrates YOLO model inference in both PyTorch and ONNX formats as part of the AI.SEE assessment.

## Project Overview

The assessment requires:
1. **PyTorch Inference**: Load and run inference with the provided YOLO model (`yolo11n.pt`)
2. **ONNX Conversion**: Convert the PyTorch model to ONNX format
3. **ONNX Inference**: Run inference using the converted ONNX model
4. **Results Comparison**: Compare results between both model formats

## 🚀 Quick Start with Docker (Recommended)

### Prerequisites
- Docker installed on your system
- Docker Compose (usually included with Docker Desktop)

### Run Complete Assessment
```bash
# Option 1: Using the convenience script
./run_docker.sh run

# Option 2: Using docker-compose directly
docker-compose up --build
```

### Available Commands
```bash
# Run complete assessment pipeline
./run_docker.sh run

# Run individual components
./run_docker.sh pytorch      # PyTorch inference only
./run_docker.sh onnx         # ONNX inference only
./run_docker.sh comparison   # Results comparison only
./run_docker.sh validation   # Environment validation only

# Development environment
./run_docker.sh dev          # Interactive development shell
./run_docker.sh jupyter      # Jupyter Lab (http://localhost:8888)

# Cleanup
./run_docker.sh clean        # Clean up Docker resources
```

## 🐍 Local Python Setup (Alternative)

### Prerequisites
- Python 3.10+
- Virtual environment support

### Installation

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate     # On Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Validate environment:**
   ```bash
   python utility/validate_environment.py
   ```

4. **Run the assessment:**
   ```bash
   # Run complete pipeline
   python pytorch_inference.py
   python onnx_inference_ultralytics.py
   python results_comparison.py
   ```

## Project Structure

```
AI.SEE-assessement/
├── input/                    # Input files
│   ├── yolo11n.pt           # PyTorch YOLO model
│   └── image-2.png          # Test image
├── output/                   # Output files
│   ├── models/              # Converted models
│   ├── results/             # Inference results
│   ├── logs/                # Execution logs
│   └── analysis/            # Comparison reports
├── utility/                 # Utility functions
├── project_notes/           # Documentation
├── requirements.txt         # Dependencies
├── Dockerfile              # Docker container definition
├── docker-compose.yml      # Docker Compose configuration
├── run_docker.sh          # Docker convenience script
├── DOCKER_GUIDE.md         # Detailed Docker documentation
└── README.md               # This file
```

## Dependencies

- **Core ML**: torch, torchvision, ultralytics
- **ONNX**: onnx, onnxruntime
- **Image Processing**: opencv-python, Pillow
- **Data Science**: numpy, matplotlib, seaborn
- **Development**: pytest, jupyter

## Usage

### 🐳 Docker Usage (Recommended)

#### Quick Start
```bash
# Run complete assessment pipeline
./run_docker.sh run

# Or using docker-compose directly
docker-compose up --build
```

#### Individual Components
```bash
# Run specific parts of the assessment
./run_docker.sh pytorch      # PyTorch inference only
./run_docker.sh onnx         # ONNX inference only  
./run_docker.sh comparison   # Results comparison only
./run_docker.sh validation   # Environment validation only
```

#### Development Environment
```bash
# Interactive development shell
./run_docker.sh dev

# Jupyter Lab (http://localhost:8888)
./run_docker.sh jupyter
```

### 🐍 Local Python Usage

#### Environment Validation
```bash
source venv/bin/activate
python utility/validate_environment.py
```

#### Running the Assessment
```bash
# Run complete pipeline
python pytorch_inference.py
python onnx_inference_ultralytics.py  
python results_comparison.py

# Or run individual components
python pytorch_inference.py              # PyTorch inference
python onnx_inference_ultralytics.py    # ONNX inference
python results_comparison.py             # Results comparison
```

## Assessment Criteria

1. **Completeness**: Successful inference in both PyTorch and ONNX
2. **Environment Setup**: Clean, isolated development environment
3. **AI Tool Usage**: Effective use of AI-assisted coding
4. **Clarity**: Clear documentation and "think aloud" process
5. **Correctness**: Accurate detection results and proper visualization

## Expected Outputs

- Annotated images with bounding boxes and labels
- Detection logs with confidence scores and coordinates
- Performance comparison between PyTorch and ONNX models
- Comprehensive documentation of the process

## Development Status

- ✅ Environment setup and validation
- ✅ PyTorch inference implementation
- ✅ ONNX conversion and inference  
- ✅ Results comparison and documentation
- ✅ Docker containerization
- ✅ Complete assessment pipeline

## 🐛 Docker Troubleshooting

### Common Issues

1. **Permission Errors**
   ```bash
   # Fix output directory permissions
   sudo chown -R $USER:$USER output/
   ```

2. **Memory Issues**
   ```bash
   # Increase Docker memory limit in Docker Desktop settings
   # Or modify docker-compose.yml resource limits
   ```

3. **Build Failures**
   ```bash
   # Clean build (no cache)
   docker-compose build --no-cache
   
   # Remove old containers
   docker-compose down --volumes
   ```

4. **Port Conflicts**
   ```bash
   # Change Jupyter port
   docker-compose run -p 8889:8888 jupyter
   ```

### Debugging Commands
```bash
# View container logs
docker-compose logs ai-see-assessment

# Access container shell
docker-compose exec ai-see-assessment bash

# Check container status
docker-compose ps

# View resource usage
docker stats ai-see-assessment
```

## 📚 Additional Documentation

- **[DOCKER_GUIDE.md](DOCKER_GUIDE.md)** - Detailed Docker documentation
- **[ASSESSMENT_SUMMARY.md](ASSESSMENT_SUMMARY.md)** - Complete assessment results
- **[project_notes/](project_notes/)** - Project planning and notes

---

*This project is part of the AI.SEE assessment demonstrating proficiency in YOLO model deployment and ONNX conversion.*
