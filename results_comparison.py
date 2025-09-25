#!/usr/bin/env python3
"""
Results Comparison for AI.SEE Assessment
This script compares PyTorch vs ONNX inference results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import pandas as pd

class ResultsComparison:
    """Compare PyTorch and ONNX inference results."""
    
    def __init__(self, output_dir="output"):
        """Initialize the results comparison."""
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / "results"
        self.analysis_dir = self.output_dir / "analysis"
        
        # Create analysis directory
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.pytorch_results = None
        self.onnx_results = None
        
    def load_results(self):
        """Load PyTorch and ONNX results from JSON files."""
        print("🔄 Loading inference results...")
        
        # Load PyTorch results
        pytorch_path = self.results_dir / "pytorch_detections.json"
        if pytorch_path.exists():
            with open(pytorch_path, 'r') as f:
                self.pytorch_results = json.load(f)
            print(f"✅ PyTorch results loaded: {len(self.pytorch_results['detections'])} detections")
        else:
            print("❌ PyTorch results not found")
            return False
        
        # Load ONNX results
        onnx_path = self.results_dir / "onnx_detections_ultralytics.json"
        if onnx_path.exists():
            with open(onnx_path, 'r') as f:
                self.onnx_results = json.load(f)
            print(f"✅ ONNX results loaded: {len(self.onnx_results['detections'])} detections")
        else:
            print("❌ ONNX results not found")
            return False
        
        return True
    
    def compare_detection_counts(self):
        """Compare the number of detections between models."""
        pytorch_count = len(self.pytorch_results['detections'])
        onnx_count = len(self.onnx_results['detections'])
        
        print(f"\n📊 Detection Count Comparison:")
        print(f"   PyTorch: {pytorch_count} detections")
        print(f"   ONNX: {onnx_count} detections")
        print(f"   Difference: {abs(pytorch_count - onnx_count)}")
        
        return {
            'pytorch_count': pytorch_count,
            'onnx_count': onnx_count,
            'difference': abs(pytorch_count - onnx_count)
        }
    
    def compare_inference_times(self):
        """Compare inference times between models."""
        pytorch_time = self.pytorch_results['model_info']['inference_time']
        onnx_time = self.onnx_results['model_info']['inference_time']
        
        speedup = pytorch_time / onnx_time if onnx_time > 0 else 0
        
        print(f"\n⏱️  Inference Time Comparison:")
        print(f"   PyTorch: {pytorch_time:.4f} seconds")
        print(f"   ONNX: {onnx_time:.4f} seconds")
        print(f"   Speedup: {speedup:.2f}x {'(ONNX faster)' if speedup > 1 else '(PyTorch faster)'}")
        
        return {
            'pytorch_time': pytorch_time,
            'onnx_time': onnx_time,
            'speedup': speedup
        }
    
    def compare_detection_confidence(self):
        """Compare confidence scores between models."""
        pytorch_confidences = [d['confidence'] for d in self.pytorch_results['detections']]
        onnx_confidences = [d['confidence'] for d in self.onnx_results['detections']]
        
        if not pytorch_confidences or not onnx_confidences:
            print("❌ No confidence scores to compare")
            return None
        
        pytorch_mean = np.mean(pytorch_confidences)
        onnx_mean = np.mean(onnx_confidences)
        
        print(f"\n🎯 Confidence Score Comparison:")
        print(f"   PyTorch mean confidence: {pytorch_mean:.4f}")
        print(f"   ONNX mean confidence: {onnx_mean:.4f}")
        print(f"   Difference: {abs(pytorch_mean - onnx_mean):.4f}")
        
        return {
            'pytorch_mean': pytorch_mean,
            'onnx_mean': onnx_mean,
            'difference': abs(pytorch_mean - onnx_mean)
        }
    
    def compare_detection_classes(self):
        """Compare detected classes between models."""
        pytorch_classes = [d['class_name'] for d in self.pytorch_results['detections']]
        onnx_classes = [d['class_name'] for d in self.onnx_results['detections']]
        
        pytorch_unique = set(pytorch_classes)
        onnx_unique = set(onnx_classes)
        
        common_classes = pytorch_unique.intersection(onnx_unique)
        pytorch_only = pytorch_unique - onnx_unique
        onnx_only = onnx_unique - pytorch_unique
        
        print(f"\n🏷️  Class Detection Comparison:")
        print(f"   PyTorch classes: {sorted(pytorch_unique)}")
        print(f"   ONNX classes: {sorted(onnx_unique)}")
        print(f"   Common classes: {sorted(common_classes)}")
        print(f"   PyTorch only: {sorted(pytorch_only)}")
        print(f"   ONNX only: {sorted(onnx_only)}")
        
        return {
            'pytorch_classes': list(pytorch_unique),
            'onnx_classes': list(onnx_unique),
            'common_classes': list(common_classes),
            'pytorch_only': list(pytorch_only),
            'onnx_only': list(onnx_only)
        }
    
    def calculate_bbox_similarity(self):
        """Calculate bounding box similarity between models."""
        if not self.pytorch_results['detections'] or not self.onnx_results['detections']:
            print("❌ No bounding boxes to compare")
            return None
        
        # Simple comparison - in a real scenario, you'd match detections by IoU
        pytorch_boxes = []
        for d in self.pytorch_results['detections']:
            bbox = d['bbox']
            pytorch_boxes.append([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']])
        
        onnx_boxes = []
        for d in self.onnx_results['detections']:
            bbox = d['bbox']
            onnx_boxes.append([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']])
        
        # Calculate average box area
        pytorch_areas = [(box[2]-box[0])*(box[3]-box[1]) for box in pytorch_boxes]
        onnx_areas = [(box[2]-box[0])*(box[3]-box[1]) for box in onnx_boxes]
        
        pytorch_mean_area = np.mean(pytorch_areas) if pytorch_areas else 0
        onnx_mean_area = np.mean(onnx_areas) if onnx_areas else 0
        
        print(f"\n📦 Bounding Box Comparison:")
        print(f"   PyTorch mean area: {pytorch_mean_area:.2f}")
        print(f"   ONNX mean area: {onnx_mean_area:.2f}")
        print(f"   Area difference: {abs(pytorch_mean_area - onnx_mean_area):.2f}")
        
        return {
            'pytorch_mean_area': pytorch_mean_area,
            'onnx_mean_area': onnx_mean_area,
            'area_difference': abs(pytorch_mean_area - onnx_mean_area)
        }
    
    def create_comparison_visualization(self):
        """Create side-by-side comparison visualization."""
        print("🔄 Creating comparison visualization...")
        
        try:
            # Load images
            import cv2
            
            # Load PyTorch result image
            pytorch_img_path = self.results_dir / "pytorch_results.png"
            onnx_img_path = self.results_dir / "onnx_results_ultralytics.png"
            
            if not pytorch_img_path.exists() or not onnx_img_path.exists():
                print("❌ Result images not found")
                return None
            
            pytorch_img = cv2.imread(str(pytorch_img_path))
            onnx_img = cv2.imread(str(onnx_img_path))
            
            if pytorch_img is None or onnx_img is None:
                print("❌ Could not load result images")
                return None
            
            # Convert BGR to RGB
            pytorch_img = cv2.cvtColor(pytorch_img, cv2.COLOR_BGR2RGB)
            onnx_img = cv2.cvtColor(onnx_img, cv2.COLOR_BGR2RGB)
            
            # Create comparison figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # PyTorch results
            ax1.imshow(pytorch_img)
            ax1.set_title('PyTorch YOLO Results', fontsize=16, fontweight='bold')
            ax1.axis('off')
            
            # ONNX results
            ax2.imshow(onnx_img)
            ax2.set_title('ONNX YOLO Results', fontsize=16, fontweight='bold')
            ax2.axis('off')
            
            plt.tight_layout()
            
            # Save comparison
            comparison_path = self.analysis_dir / "pytorch_vs_onnx_comparison.png"
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            print(f"✅ Comparison visualization saved to: {comparison_path}")
            
            plt.show()
            return comparison_path
            
        except Exception as e:
            print(f"❌ Comparison visualization failed: {e}")
            return None
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report."""
        print("🔄 Generating comparison report...")
        
        # Run all comparisons
        detection_counts = self.compare_detection_counts()
        inference_times = self.compare_inference_times()
        confidence_scores = self.compare_detection_confidence()
        class_comparison = self.compare_detection_classes()
        bbox_comparison = self.calculate_bbox_similarity()
        
        # Create comprehensive report
        report_data = {
            'comparison_summary': {
                'timestamp': datetime.now().isoformat(),
                'pytorch_model': self.pytorch_results['model_info']['model_path'],
                'onnx_model': self.onnx_results['model_info']['model_path'],
                'image_path': self.pytorch_results['image_info']['image_path']
            },
            'detection_counts': detection_counts,
            'inference_times': inference_times,
            'confidence_scores': confidence_scores,
            'class_comparison': class_comparison,
            'bbox_comparison': bbox_comparison
        }
        
        # Save JSON report
        json_path = self.analysis_dir / "comparison_report.json"
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Create markdown report
        md_path = self.analysis_dir / "comparison_report.md"
        with open(md_path, 'w') as f:
            f.write("# AI.SEE Assessment - PyTorch vs ONNX Comparison Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **PyTorch Detections:** {detection_counts['pytorch_count']}\n")
            f.write(f"- **ONNX Detections:** {detection_counts['onnx_count']}\n")
            f.write(f"- **Detection Difference:** {detection_counts['difference']}\n")
            f.write(f"- **PyTorch Inference Time:** {inference_times['pytorch_time']:.4f}s\n")
            f.write(f"- **ONNX Inference Time:** {inference_times['onnx_time']:.4f}s\n")
            f.write(f"- **Speedup:** {inference_times['speedup']:.2f}x\n\n")
            
            f.write("## Detailed Analysis\n\n")
            f.write("### Detection Counts\n")
            f.write(f"- PyTorch: {detection_counts['pytorch_count']} detections\n")
            f.write(f"- ONNX: {detection_counts['onnx_count']} detections\n")
            f.write(f"- Difference: {detection_counts['difference']}\n\n")
            
            f.write("### Performance\n")
            f.write(f"- PyTorch time: {inference_times['pytorch_time']:.4f} seconds\n")
            f.write(f"- ONNX time: {inference_times['onnx_time']:.4f} seconds\n")
            f.write(f"- Speedup: {inference_times['speedup']:.2f}x\n\n")
            
            if confidence_scores:
                f.write("### Confidence Scores\n")
                f.write(f"- PyTorch mean: {confidence_scores['pytorch_mean']:.4f}\n")
                f.write(f"- ONNX mean: {confidence_scores['onnx_mean']:.4f}\n")
                f.write(f"- Difference: {confidence_scores['difference']:.4f}\n\n")
            
            if class_comparison:
                f.write("### Class Detection\n")
                f.write(f"- PyTorch classes: {', '.join(class_comparison['pytorch_classes'])}\n")
                f.write(f"- ONNX classes: {', '.join(class_comparison['onnx_classes'])}\n")
                f.write(f"- Common classes: {', '.join(class_comparison['common_classes'])}\n")
                f.write(f"- PyTorch only: {', '.join(class_comparison['pytorch_only'])}\n")
                f.write(f"- ONNX only: {', '.join(class_comparison['onnx_only'])}\n\n")
            
            f.write("## Conclusion\n\n")
            if detection_counts['difference'] == 0:
                f.write("✅ **Perfect Match:** Both models detected the same number of objects.\n")
            else:
                f.write(f"⚠️ **Detection Difference:** Models detected different numbers of objects ({detection_counts['difference']} difference).\n")
            
            if inference_times['speedup'] > 1:
                f.write(f"🚀 **ONNX Faster:** ONNX model is {inference_times['speedup']:.2f}x faster than PyTorch.\n")
            else:
                f.write(f"🐌 **PyTorch Faster:** PyTorch model is {1/inference_times['speedup']:.2f}x faster than ONNX.\n")
        
        print(f"✅ Comparison report saved to: {md_path}")
        print(f"✅ JSON data saved to: {json_path}")
        
        return md_path, json_path
    
    def run_full_comparison(self):
        """Run the complete comparison analysis."""
        print("🚀 Starting Results Comparison Analysis")
        print("=" * 60)
        
        # Load results
        if not self.load_results():
            return False
        
        # Run comparisons
        self.generate_comparison_report()
        self.create_comparison_visualization()
        
        print("\n✅ Results comparison completed successfully!")
        return True

def main():
    """Main function to run results comparison."""
    comparison = ResultsComparison()
    success = comparison.run_full_comparison()
    
    if success:
        print("\n🎉 Results comparison completed successfully!")
        print("Check the 'output/analysis/' directory for comparison reports.")
    else:
        print("\n❌ Results comparison failed!")

if __name__ == "__main__":
    main()
