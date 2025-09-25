# AI.SEE Assessment - Docker Guide

This guide explains how to run the AI.SEE assessment using Docker containers.

## 🐳 Quick Start

### Prerequisites
- Docker installed on your system
- Docker Compose (usually included with Docker Desktop)

### Run Complete Assessment
```bash
# Build and run the full assessment pipeline
docker-compose up --build

# Run in detached mode (background)
docker-compose up -d --build

# View logs
docker-compose logs -f
```

## 📋 Available Services

### 1. Main Assessment Service (`ai-see-assessment`)
Runs the complete AI.SEE assessment pipeline:
- Environment validation
- PyTorch inference
- ONNX conversion
- ONNX inference  
- Results comparison

```bash
# Run only the main service
docker-compose up ai-see-assessment

# Run with custom command
docker-compose run ai-see-assessment python pytorch_inference.py
```

### 2. Jupyter Development Service (`jupyter`)
Interactive development environment with Jupyter Lab:

```bash
# Start Jupyter service
docker-compose --profile development up jupyter

# Access at: http://localhost:8888
```

### 3. Development Service (`development`)
Interactive shell for development and testing:

```bash
# Start development environment
docker-compose --profile development up development

# Access interactive shell
docker-compose --profile development run development bash
```

## 🛠️ Development Commands

### Run Individual Scripts
```bash
# PyTorch inference only
docker-compose run ai-see-assessment python pytorch_inference.py

# ONNX inference only
docker-compose run ai-see-assessment python onnx_inference_ultralytics.py

# Results comparison only
docker-compose run ai-see-assessment python results_comparison.py

# Environment validation
docker-compose run ai-see-assessment python utility/validate_environment.py
```

### Interactive Development
```bash
# Start development container
docker-compose --profile development up development

# Access container shell
docker-compose --profile development exec development bash

# Run commands inside container
docker-compose --profile development exec development python pytorch_inference.py
```

## 📁 Volume Mounts

The Docker setup includes the following volume mounts:

- `./input:/app/input:ro` - Input files (read-only)
- `./output:/app/output` - Output results (read-write)
- `./project_notes:/app/project_notes:ro` - Documentation (read-only)

## 🔧 Customization

### Environment Variables
```bash
# Set custom environment variables
docker-compose run -e CUDA_VISIBLE_DEVICES=0 ai-see-assessment python pytorch_inference.py
```

### Resource Limits
Modify `docker-compose.yml` to adjust resource limits:
```yaml
deploy:
  resources:
    limits:
      memory: 8G  # Increase memory limit
      cpus: '4.0'  # Set CPU limit
```

### Custom Commands
```bash
# Run custom Python script
docker-compose run ai-see-assessment python your_script.py

# Run with custom arguments
docker-compose run ai-see-assessment python pytorch_inference.py --conf 0.3
```

## 🐛 Troubleshooting

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

### Debugging

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

## 📊 Output Files

After running the assessment, check the `output/` directory:

```
output/
├── models/
│   ├── yolo11n.onnx              # Converted ONNX model
│   └── onnx_model_info.json      # Model metadata
├── results/
│   ├── pytorch_results.png       # PyTorch visualization
│   ├── pytorch_detections.json   # PyTorch results
│   ├── onnx_results_ultralytics.png  # ONNX visualization
│   └── onnx_detections_ultralytics.json  # ONNX results
├── logs/
│   ├── pytorch_inference.log     # PyTorch execution log
│   └── onnx_inference_ultralytics.log  # ONNX execution log
└── analysis/
    ├── comparison_report.md       # Detailed comparison
    ├── comparison_report.json    # Comparison data
    └── pytorch_vs_onnx_comparison.png  # Side-by-side visualization
```

## 🚀 Production Deployment

### Build Production Image
```bash
# Build optimized production image
docker build -t ai-see-assessment:latest .

# Run production container
docker run -v $(pwd)/input:/app/input:ro \
           -v $(pwd)/output:/app/output \
           ai-see-assessment:latest
```

### Multi-Architecture Build
```bash
# Build for multiple architectures
docker buildx build --platform linux/amd64,linux/arm64 -t ai-see-assessment:multi-arch .
```

## 🔒 Security Considerations

- Container runs as non-root user (`appuser`)
- Input files mounted as read-only
- No unnecessary network exposure
- Resource limits applied
- Minimal base image used

## 📝 Notes

- The container uses CPU-only inference (no GPU support by default)
- All results are saved to the mounted `output/` directory
- The container includes all necessary dependencies
- Jupyter service is optional and only available in development profile
