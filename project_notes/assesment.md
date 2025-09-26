# Assessment for AI.SEE
## Context
You will receive two files:
1. yolo1in.pt a YOLO model in PyTorch format.
2. image.png a sample image to test against.

## Your assignment has two main tasks
1. Predict (image.png) using the given YOLO model,
2. Convert the yololin.pt model to ONNX and then use the converted ONNX model to run inference on the same image.png.

## Detailed Instructions
### Environment Setup from Scratch
1. Create a new development environment (e.g., a fresh Python virtual environment or Docker container).
2. Install all libraries required to run the PyTorch model and to perform the ONNX conversion (e.g., torch, onnx, onnxruntime, etc.).

### Inference with the PyTorch Model
1. Load the yolo1in.pt model in PyTorch
2. Run inference on image.png and output the results, such as bounding box coordinates, labels, and/or a labeled output image showing detections

### Convert the Model to ONNX
1. Convert the PyTorch model (yolo1in.pt) to ONNX format
2. Ensure the ONNX model is valid by using it for inference in your environment

### Inference with the ONNX Model
1. Once the conversion is successful, use the ONNX model to run inference on image.png again
2. Output the results in a human-readable format (e.g., console logs, bounding box overlays, etc.).

### Evaluation Criteria for AI.SEE
1. Completeness: Did you successfully run inference in both PyTorch and ONNX?
2. Environment Setup: How well did you configure everything from scratch?
3. AI Tool Usage: How effectively you use AI-assisted coding
4. Clarity of Explanation: The detail and clarity in your "think aloud" process
5. Correctness of Outputs: Are the bounding boxes/labels (or any other output) sensible?