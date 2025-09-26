#!/bin/bash

# YOLO Assessment Runner Script
# This script provides easy commands to run the assessment in Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to check if required files exist
check_files() {
    local missing_files=()
    
    if [ ! -f "input/yolo11n.pt" ]; then
        missing_files+=("input/yolo11n.pt")
    fi
    
    if [ ! -f "input/image-2.png" ]; then
        missing_files+=("input/image-2.png")
    fi
    
    if [ ${#missing_files[@]} -ne 0 ]; then
        print_error "Missing required files:"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        exit 1
    fi
}

# Function to create necessary directories
create_directories() {
    mkdir -p output utility project_notes
    print_status "Created necessary directories"
}

# Function to build Docker image
build_image() {
    print_status "Building Docker image..."
    docker-compose build
    print_success "Docker image built successfully"
}

# Function to run full assessment
run_full_assessment() {
    print_status "Running full YOLO assessment pipeline..."
    docker-compose up yolo-assessment
    print_success "Full assessment completed"
}

# Function to run individual steps
run_step() {
    local step=$1
    case $step in
        "pytorch")
            print_status "Running PyTorch inference..."
            docker-compose --profile pytorch up pytorch-inference
            ;;
        "conversion")
            print_status "Running ONNX conversion..."
            docker-compose --profile conversion up onnx-conversion
            ;;
        "onnx")
            print_status "Running ONNX inference..."
            docker-compose --profile onnx up onnx-inference
            ;;
        "comparison")
            print_status "Running results comparison..."
            docker-compose --profile comparison up results-comparison
            ;;
        *)
            print_error "Unknown step: $step"
            print_status "Available steps: pytorch, conversion, onnx, comparison"
            exit 1
            ;;
    esac
}

# Function to start development environment
start_dev() {
    print_status "Starting development environment..."
    docker-compose --profile dev up -d dev
    print_success "Development environment started"
    print_status "Connect with: docker exec -it yolo-dev bash"
}

# Function to start Jupyter notebook
start_jupyter() {
    print_status "Starting Jupyter notebook..."
    docker-compose --profile jupyter up -d jupyter
    print_success "Jupyter notebook started"
    print_status "Access at: http://localhost:8888"
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker containers..."
    docker-compose down
    docker system prune -f
    print_success "Cleanup completed"
}

# Function to show help
show_help() {
    echo "YOLO Assessment Runner"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build              Build Docker image"
    echo "  run                Run full assessment pipeline"
    echo "  step <name>        Run individual step (pytorch|conversion|onnx|comparison)"
    echo "  dev                Start development environment"
    echo "  jupyter            Start Jupyter notebook"
    echo "  cleanup            Clean up Docker containers and images"
    echo "  help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 run"
    echo "  $0 step pytorch"
    echo "  $0 dev"
    echo "  $0 jupyter"
}

# Main script logic
main() {
    # Check if Docker is running
    check_docker
    
    # Create necessary directories
    create_directories
    
    # Check if required files exist
    check_files
    
    # Parse command line arguments
    case "${1:-help}" in
        "build")
            build_image
            ;;
        "run")
            build_image
            run_full_assessment
            ;;
        "step")
            if [ -z "$2" ]; then
                print_error "Step name required"
                show_help
                exit 1
            fi
            build_image
            run_step "$2"
            ;;
        "dev")
            build_image
            start_dev
            ;;
        "jupyter")
            build_image
            start_jupyter
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function
main "$@"
