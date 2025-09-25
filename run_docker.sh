#!/bin/bash
# AI.SEE Assessment - Docker Runner Script

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
    if [ ! -f "input/yolo11n.pt" ]; then
        print_error "Model file not found: input/yolo11n.pt"
        exit 1
    fi
    
    if [ ! -f "input/image-2.png" ]; then
        print_error "Image file not found: input/image-2.png"
        exit 1
    fi
    
    print_success "Required files found"
}

# Function to create output directory
create_output_dir() {
    mkdir -p output/{models,results,logs,analysis}
    print_success "Output directories created"
}

# Function to run full assessment
run_assessment() {
    print_status "Starting AI.SEE Assessment..."
    print_status "This will run the complete pipeline: PyTorch → ONNX → Comparison"
    
    docker-compose up --build ai-see-assessment
    
    if [ $? -eq 0 ]; then
        print_success "Assessment completed successfully!"
        print_status "Check the 'output/' directory for results"
    else
        print_error "Assessment failed!"
        exit 1
    fi
}

# Function to run individual components
run_pytorch() {
    print_status "Running PyTorch inference only..."
    docker-compose run --rm ai-see-assessment python pytorch_inference.py
}

run_onnx() {
    print_status "Running ONNX inference only..."
    docker-compose run --rm ai-see-assessment python onnx_inference_ultralytics.py
}

run_comparison() {
    print_status "Running results comparison..."
    docker-compose run --rm ai-see-assessment python results_comparison.py
}

run_validation() {
    print_status "Running environment validation..."
    docker-compose run --rm ai-see-assessment python utility/validate_environment.py
}

# Function to start development environment
start_dev() {
    print_status "Starting development environment..."
    docker-compose --profile development up development
}

# Function to start Jupyter
start_jupyter() {
    print_status "Starting Jupyter Lab..."
    print_status "Access at: http://localhost:8888"
    docker-compose --profile development up jupyter
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose down --volumes --remove-orphans
    docker system prune -f
    print_success "Cleanup completed"
}

# Function to show help
show_help() {
    echo "AI.SEE Assessment - Docker Runner"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  run, assessment    Run complete assessment pipeline"
    echo "  pytorch           Run PyTorch inference only"
    echo "  onnx              Run ONNX inference only"
    echo "  comparison        Run results comparison only"
    echo "  validation        Run environment validation only"
    echo "  dev, development   Start development environment"
    echo "  jupyter           Start Jupyter Lab"
    echo "  clean, cleanup    Clean up Docker resources"
    echo "  help, -h, --help  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 run            # Run complete assessment"
    echo "  $0 pytorch        # Run only PyTorch inference"
    echo "  $0 dev            # Start development environment"
    echo "  $0 jupyter        # Start Jupyter Lab"
}

# Main script logic
main() {
    # Check prerequisites
    check_docker
    check_files
    create_output_dir
    
    # Parse command
    case "${1:-run}" in
        "run"|"assessment")
            run_assessment
            ;;
        "pytorch")
            run_pytorch
            ;;
        "onnx")
            run_onnx
            ;;
        "comparison")
            run_comparison
            ;;
        "validation")
            run_validation
            ;;
        "dev"|"development")
            start_dev
            ;;
        "jupyter")
            start_jupyter
            ;;
        "clean"|"cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
