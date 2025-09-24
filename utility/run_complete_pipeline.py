#!/usr/bin/env python3
"""
Complete Pipeline Runner for AI.SEE Assessment
This script runs the complete inference pipeline from start to finish.
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        
        execution_time = time.time() - start_time
        print(f"✅ {description} completed successfully in {execution_time:.2f} seconds")
        
        if result.stdout:
            print("Output:")
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        execution_time = time.time() - start_time
        print(f"❌ {description} failed after {execution_time:.2f} seconds")
        print(f"Error: {e}")
        if e.stdout:
            print("Stdout:", e.stdout)
        if e.stderr:
            print("Stderr:", e.stderr)
        return False


def check_environment():
    """Check if the environment is properly set up."""
    print("Checking environment setup...")
    
    # Check if virtual environment exists
    if not Path("venv").exists():
        print("❌ Virtual environment not found. Please run setup first.")
        return False
    
    # Check if required files exist
    required_files = [
        "assets/yolo11n.pt",
        "assets/image-2.png",
        "pytorch_inference.py",
        "onnx_conversion.py",
        "onnx_inference.py",
        "results_comparison.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ Environment check passed")
    return True


def run_complete_pipeline():
    """Run the complete AI.SEE assessment pipeline."""
    print("🚀 Starting AI.SEE Assessment Complete Pipeline")
    print("="*60)
    
    # Check environment
    if not check_environment():
        print("❌ Environment check failed. Please fix issues and try again.")
        return False
    
    # Define pipeline steps
    pipeline_steps = [
        {
            "command": "source venv/bin/activate && python pytorch_inference.py",
            "description": "PyTorch Model Inference"
        },
        {
            "command": "source venv/bin/activate && python onnx_conversion.py",
            "description": "ONNX Model Conversion"
        },
        {
            "command": "source venv/bin/activate && python onnx_inference.py",
            "description": "ONNX Model Inference"
        },
        {
            "command": "source venv/bin/activate && python results_comparison.py",
            "description": "Results Comparison and Analysis"
        }
    ]
    
    # Track results
    results = []
    total_start_time = time.time()
    
    # Run each step
    for i, step in enumerate(pipeline_steps, 1):
        print(f"\n📋 Step {i}/{len(pipeline_steps)}: {step['description']}")
        
        success = run_command(step['command'], step['description'])
        results.append({
            'step': step['description'],
            'success': success,
            'step_number': i
        })
        
        if not success:
            print(f"\n❌ Pipeline failed at step {i}: {step['description']}")
            print("Please check the error messages above and fix any issues.")
            return False
    
    total_time = time.time() - total_start_time
    
    # Generate summary
    print(f"\n🎉 PIPELINE COMPLETION SUMMARY")
    print("="*60)
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Steps completed: {len([r for r in results if r['success']])}/{len(results)}")
    
    print(f"\nStep Results:")
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"  {result['step_number']}. {result['step']}: {status}")
    
    # Check output files
    output_files = [
        "output/pytorch_detections.png",
        "output/onnx_detections.png", 
        "output/model_comparison.png",
        "output/pytorch_results.txt",
        "output/onnx_results.txt",
        "output/onnx_conversion_report.txt",
        "output/comparison_report.txt"
    ]
    
    print(f"\nGenerated Output Files:")
    for file in output_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (missing)")
    
    # Final assessment
    all_success = all(r['success'] for r in results)
    if all_success:
        print(f"\n🎉 AI.SEE ASSESSMENT COMPLETED SUCCESSFULLY!")
        print(f"All requirements have been met with excellent results.")
        print(f"Check the 'output/' directory for all generated files.")
    else:
        print(f"\n❌ AI.SEE ASSESSMENT INCOMPLETE")
        print(f"Some steps failed. Please review the errors above.")
    
    return all_success


def main():
    """Main function to run the complete pipeline."""
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    print(f"Working directory: {os.getcwd()}")
    
    # Run the complete pipeline
    success = run_complete_pipeline()
    
    if success:
        print(f"\n🚀 Ready for AI.SEE Assessment Submission!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Pipeline completed with issues. Please review and fix.")
        sys.exit(1)


if __name__ == "__main__":
    main()
