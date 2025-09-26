# Docker Setup for YOLO Assessment

This document provides instructions for running the YOLO assessment in a Dockerized environment.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB RAM available for Docker
- 2GB free disk space

## Quick Start

### 1. Build and Run Full Assessment

```bash
# Make the script executable (if not already)
chmod +x run_assessment.sh

# Run the complete assessment pipeline
./run_assessment.sh run
```

### 2. Run Individual Steps

```bash
# Run only PyTorch inference
./run_assessment.sh step pytorch

# Run only ONNX conversion
./run_assessment.sh step conversion

# Run only ONNX inference
./run_assessment.sh step onnx

# Run only results comparison
./run_assessment.sh step comparison
```

## Docker Services

### Main Services

#### `yolo-assessment` (Default)
Runs the complete assessment pipeline:
- PyTorch inference
- ONNX conversion
- ONNX inference
- Results comparison

```bash
docker-compose up yolo-assessment
```

#### Individual Step Services
Run specific steps using profiles:

```bash
# PyTorch inference only
docker-compose --profile pytorch up pytorch-inference

# ONNX conversion only
docker-compose --profile conversion up onnx-conversion

# ONNX inference only
docker-compose --profile onnx up onnx-inference

# Results comparison only
docker-compose --profile comparison up results-comparison
```

### Development Services

#### `dev` - Development Environment
Interactive shell for development and debugging:

```bash
# Start development container
./run_assessment.sh dev

# Connect to the container
docker exec -it yolo-dev bash
```

#### `jupyter` - Jupyter Notebook
Interactive Jupyter notebook environment:

```bash
# Start Jupyter notebook
./run_assessment.sh jupyter

# Access at http://localhost:8888
```

## Manual Docker Commands

### Build Image
```bash
docker-compose build
```

### Run Full Pipeline
```bash
docker-compose up yolo-assessment
```

### Run with Custom Command
```bash
docker-compose run --rm yolo-assessment python pytorch_inference.py
```

### Access Container Shell
```bash
docker-compose run --rm yolo-assessment /bin/bash
```

## Volume Mounts

The Docker setup uses the following volume mounts:

- `./input:/app/input:ro` - Read-only access to input files
- `./output:/app/output` - Write access to output directory
- `./project_notes:/app/project_notes:ro` - Read-only access to documentation
- `./utility:/app/utility:ro` - Read-only access to utility functions

## Environment Variables

- `PYTHONPATH=/app` - Python path configuration
- `CUDA_VISIBLE_DEVICES=""` - Force CPU usage (no GPU)

## File Structure in Container

```
/app/
├── input/              # Mounted input files
├── output/             # Mounted output directory
├── utility/            # Mounted utility functions
├── project_notes/      # Mounted documentation
├── pytorch_inference.py
├── onnx_conversion.py
├── onnx_inference.py
├── results_comparison.py
└── requirements.txt
```

## Troubleshooting

### Common Issues

#### 1. Permission Denied
```bash
# Fix script permissions
chmod +x run_assessment.sh

# Fix output directory permissions
sudo chown -R $USER:$USER output/
```

#### 2. Docker Not Running
```bash
# Start Docker service
sudo systemctl start docker

# Check Docker status
docker info
```

#### 3. Out of Memory
```bash
# Increase Docker memory limit in Docker Desktop
# Or use smaller batch sizes in the scripts
```

#### 4. Port Already in Use (Jupyter)
```bash
# Stop existing Jupyter container
docker-compose --profile jupyter down

# Or use different port
docker-compose --profile jupyter up -d jupyter
```

### Debugging

#### View Container Logs
```bash
# View logs for specific service
docker-compose logs yolo-assessment

# Follow logs in real-time
docker-compose logs -f yolo-assessment
```

#### Inspect Container
```bash
# List running containers
docker-compose ps

# Inspect container details
docker inspect yolo-assessment
```

#### Access Container Filesystem
```bash
# Copy files from container
docker cp yolo-assessment:/app/output/results.txt ./local_results.txt

# Copy files to container
docker cp ./local_file.txt yolo-assessment:/app/
```

## Performance Optimization

### For Faster Builds
```bash
# Use build cache
docker-compose build --no-cache

# Use multi-stage builds (advanced)
# See Dockerfile for optimization
```

### For Better Performance
```bash
# Increase Docker memory allocation
# Use SSD storage for Docker volumes
# Enable Docker BuildKit
export DOCKER_BUILDKIT=1
```

## Security Considerations

- Input files are mounted read-only
- No sensitive data in Docker images
- Container runs as non-root user
- Network access is limited

## Cleanup

### Remove Containers and Images
```bash
# Clean up everything
./run_assessment.sh cleanup

# Or manually
docker-compose down
docker system prune -f
```

### Remove Volumes
```bash
# Remove all volumes (WARNING: deletes data)
docker-compose down -v
docker volume prune -f
```

## Advanced Usage

### Custom Dockerfile
```dockerfile
# Extend the base image
FROM yolo-assessment:latest

# Add custom dependencies
RUN pip install custom-package

# Override default command
CMD ["python", "custom_script.py"]
```

### Environment-Specific Configuration
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  yolo-assessment:
    environment:
      - CUDA_VISIBLE_DEVICES=0  # Enable GPU
    volumes:
      - /path/to/gpu/models:/app/models
```

## Support

For issues with the Docker setup:

1. Check the troubleshooting section
2. Review Docker logs
3. Verify file permissions
4. Ensure sufficient system resources

## Contributing

When modifying the Docker setup:

1. Test with `./run_assessment.sh build`
2. Verify all services work
3. Update documentation
4. Test cleanup procedures
