# AI.SEE Assessment Todo List

## Environment Setup
- [ ] Create Python virtual environment
- [ ] Create requirements.txt with all necessary dependencies
- [ ] Install PyTorch and related libraries
- [ ] Install ONNX and ONNXRuntime
- [ ] Install OpenCV for image processing
- [ ] Install Ultralytics YOLO package
- [ ] Verify all installations work correctly

## PyTorch Model Inference
- [ ] Load yolo11n.pt model using Ultralytics YOLO API
- [ ] Verify model architecture and input requirements
- [ ] Check model metadata for class names and configuration
- [ ] Load and preprocess image-2.png to match model input format
- [ ] Run inference to obtain raw detection outputs
- [ ] Apply non-maximum suppression to filter overlapping detections
- [ ] Extract bounding boxes, class labels, and confidence scores
- [ ] Create annotated output image with detection overlays
- [ ] Log detailed results to console for analysis
- [ ] Save PyTorch inference results to output folder

## ONNX Conversion
- [ ] Use Ultralytics export functionality for PyTorch to ONNX conversion
- [ ] Specify appropriate input dimensions and batch size
- [ ] Validate converted model using ONNX model checker
- [ ] Test basic inference to ensure conversion success
- [ ] Compare model input/output shapes between PyTorch and ONNX versions
- [ ] Verify that class names and model metadata are preserved
- [ ] Save converted ONNX model to output folder

## ONNX Model Inference
- [ ] Load the converted ONNX model
- [ ] Run inference on image-2.png using ONNX model
- [ ] Apply same post-processing as PyTorch model
- [ ] Extract bounding boxes, class labels, and confidence scores
- [ ] Create annotated output image with detection overlays
- [ ] Log detailed results to console for analysis
- [ ] Save ONNX inference results to output folder

## Results Comparison and Analysis
- [ ] Compare detection results between PyTorch and ONNX models
- [ ] Analyze performance differences (speed, accuracy)
- [ ] Document any discrepancies in outputs
- [ ] Create comparison report
- [ ] Generate side-by-side visualization of results

## Documentation and Cleanup
- [ ] Create comprehensive README with setup instructions
- [ ] Document all code with clear comments
- [ ] Create execution log with step-by-step process
- [ ] Document decision rationale for technical choices
- [ ] Record performance observations and comparisons
- [ ] Note lessons learned and potential improvements
- [ ] Organize all output files in proper folder structure
- [ ] Create final assessment report

## Quality Assurance
- [ ] Verify all bounding boxes and labels are sensible
- [ ] Ensure consistent preprocessing between both pipelines
- [ ] Test error handling for edge cases
- [ ] Validate output file formats and locations
- [ ] Review code for PEP 8 compliance
- [ ] Test complete workflow from start to finish

## Final Deliverables
- [ ] Working PyTorch inference script
- [ ] Working ONNX conversion script
- [ ] Working ONNX inference script
- [ ] Comparison analysis report
- [ ] All output images and results
- [ ] Complete documentation
- [ ] Requirements.txt for environment reproduction
