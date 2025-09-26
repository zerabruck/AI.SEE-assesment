#!/usr/bin/env python3
"""
Results Comparison Script
AI.SEE Assessment - Step 5: PyTorch vs ONNX Results Comparison

This script compares the results from PyTorch and ONNX inference to analyze
performance, accuracy, and differences between the two model formats.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/comparison.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ResultsComparator:
    """Class to compare PyTorch and ONNX inference results."""
    
    def __init__(self, pytorch_results_file: str, onnx_results_file: str):
        """
        Initialize the results comparator.
        
        Args:
            pytorch_results_file: Path to PyTorch results file
            onnx_results_file: Path to ONNX results file
        """
        self.pytorch_results_file = pytorch_results_file
        self.onnx_results_file = onnx_results_file
        self.pytorch_data = None
        self.onnx_data = None
        
        self._load_results()
    
    def _load_results(self) -> None:
        """Load results from both PyTorch and ONNX inference."""
        try:
            logger.info("Loading PyTorch results...")
            self.pytorch_data = self._parse_results_file(self.pytorch_results_file)
            
            logger.info("Loading ONNX results...")
            self.onnx_data = self._parse_results_file(self.onnx_results_file)
            
            logger.info("Results loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            raise
    
    def _parse_results_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse results file and extract detection data.
        
        Args:
            file_path: Path to the results file
            
        Returns:
            Dictionary containing parsed results
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract basic information
            lines = content.split('\n')
            inference_time = None
            num_detections = None
            confidence_threshold = None
            
            for line in lines:
                if 'Inference Time:' in line:
                    inference_time = float(line.split(':')[1].strip().split()[0])
                elif 'Number of Detections:' in line:
                    num_detections = int(line.split(':')[1].strip())
                elif 'Confidence Threshold:' in line:
                    confidence_threshold = float(line.split(':')[1].strip())
            
            # Extract detections
            detections = []
            in_detections_section = False
            
            for line in lines:
                if 'Detections:' in line:
                    in_detections_section = True
                    continue
                elif in_detections_section and line.startswith('Detection'):
                    # Parse detection data
                    detection_data = {}
                    detection_lines = []
                    i = lines.index(line)
                    
                    # Collect all lines for this detection
                    j = i + 1
                    while j < len(lines) and not lines[j].startswith('Detection') and lines[j].strip():
                        detection_lines.append(lines[j])
                        j += 1
                    
                    # Parse detection information
                    for det_line in detection_lines:
                        if 'Class:' in det_line:
                            detection_data['class'] = det_line.split(':')[1].strip()
                        elif 'Confidence:' in det_line:
                            detection_data['confidence'] = float(det_line.split(':')[1].strip())
                        elif 'Bounding Box:' in det_line:
                            bbox_str = det_line.split(':')[1].strip()
                            bbox_str = bbox_str.replace('[', '').replace(']', '')
                            bbox = [float(x.strip()) for x in bbox_str.split(',')]
                            detection_data['bbox'] = bbox
                        elif 'Width:' in det_line and 'Height:' in det_line:
                            parts = det_line.split(',')
                            width = float(parts[0].split(':')[1].strip())
                            height = float(parts[1].split(':')[1].strip())
                            detection_data['width'] = width
                            detection_data['height'] = height
                    
                    if detection_data:
                        detections.append(detection_data)
            
            return {
                'inference_time': inference_time,
                'num_detections': num_detections,
                'confidence_threshold': confidence_threshold,
                'detections': detections
            }
            
        except Exception as e:
            logger.error(f"Failed to parse results file {file_path}: {e}")
            raise
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def compare_detections(self, iou_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Compare detections between PyTorch and ONNX models.
        
        Args:
            iou_threshold: IoU threshold for considering detections as matches
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            logger.info("Comparing detections between PyTorch and ONNX models...")
            
            pytorch_detections = self.pytorch_data['detections']
            onnx_detections = self.onnx_data['detections']
            
            # Initialize comparison results
            comparison = {
                'pytorch_detections': len(pytorch_detections),
                'onnx_detections': len(onnx_detections),
                'matched_detections': 0,
                'pytorch_only': 0,
                'onnx_only': 0,
                'class_comparison': {},
                'confidence_comparison': [],
                'bbox_comparison': [],
                'iou_threshold': iou_threshold
            }
            
            # Track matched detections
            pytorch_matched = [False] * len(pytorch_detections)
            onnx_matched = [False] * len(onnx_detections)
            
            # Compare each PyTorch detection with ONNX detections
            for i, pt_det in enumerate(pytorch_detections):
                best_iou = 0
                best_match_idx = -1
                
                for j, onnx_det in enumerate(onnx_detections):
                    if onnx_matched[j]:
                        continue
                    
                    # Check if same class
                    if pt_det['class'] == onnx_det['class']:
                        iou = self.calculate_iou(pt_det['bbox'], onnx_det['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_match_idx = j
                
                if best_iou >= iou_threshold:
                    # Match found
                    pytorch_matched[i] = True
                    onnx_matched[best_match_idx] = True
                    comparison['matched_detections'] += 1
                    
                    # Store comparison data
                    comparison['confidence_comparison'].append({
                        'pytorch_confidence': pt_det['confidence'],
                        'onnx_confidence': onnx_detections[best_match_idx]['confidence'],
                        'class': pt_det['class'],
                        'iou': best_iou
                    })
                    
                    comparison['bbox_comparison'].append({
                        'pytorch_bbox': pt_det['bbox'],
                        'onnx_bbox': onnx_detections[best_match_idx]['bbox'],
                        'class': pt_det['class'],
                        'iou': best_iou
                    })
                else:
                    comparison['pytorch_only'] += 1
            
            # Count ONNX-only detections
            comparison['onnx_only'] = sum(1 for matched in onnx_matched if not matched)
            
            # Class-wise comparison
            pytorch_classes = [det['class'] for det in pytorch_detections]
            onnx_classes = [det['class'] for det in onnx_detections]
            
            all_classes = set(pytorch_classes + onnx_classes)
            for class_name in all_classes:
                pytorch_count = pytorch_classes.count(class_name)
                onnx_count = onnx_classes.count(class_name)
                comparison['class_comparison'][class_name] = {
                    'pytorch': pytorch_count,
                    'onnx': onnx_count,
                    'difference': onnx_count - pytorch_count
                }
            
            logger.info(f"Comparison completed: {comparison['matched_detections']} matches found")
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare detections: {e}")
            raise
    
    def create_comparison_visualization(self, comparison: Dict[str, Any], 
                                      output_path: str) -> None:
        """
        Create visualization comparing PyTorch and ONNX results.
        
        Args:
            comparison: Comparison results dictionary
            output_path: Path to save the visualization
        """
        try:
            logger.info("Creating comparison visualization...")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('PyTorch vs ONNX Model Comparison', fontsize=16, fontweight='bold')
            
            # 1. Detection count comparison
            ax1 = axes[0, 0]
            models = ['PyTorch', 'ONNX']
            counts = [comparison['pytorch_detections'], comparison['onnx_detections']]
            colors = ['#1f77b4', '#ff7f0e']
            
            bars = ax1.bar(models, counts, color=colors, alpha=0.7)
            ax1.set_title('Total Detections Count')
            ax1.set_ylabel('Number of Detections')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            # 2. Class-wise comparison
            ax2 = axes[0, 1]
            classes = list(comparison['class_comparison'].keys())
            pytorch_counts = [comparison['class_comparison'][cls]['pytorch'] for cls in classes]
            onnx_counts = [comparison['class_comparison'][cls]['onnx'] for cls in classes]
            
            x = np.arange(len(classes))
            width = 0.35
            
            ax2.bar(x - width/2, pytorch_counts, width, label='PyTorch', color='#1f77b4', alpha=0.7)
            ax2.bar(x + width/2, onnx_counts, width, label='ONNX', color='#ff7f0e', alpha=0.7)
            
            ax2.set_title('Class-wise Detection Count')
            ax2.set_ylabel('Number of Detections')
            ax2.set_xlabel('Classes')
            ax2.set_xticks(x)
            ax2.set_xticklabels(classes, rotation=45, ha='right')
            ax2.legend()
            
            # 3. Confidence comparison
            ax3 = axes[1, 0]
            if comparison['confidence_comparison']:
                pytorch_conf = [comp['pytorch_confidence'] for comp in comparison['confidence_comparison']]
                onnx_conf = [comp['onnx_confidence'] for comp in comparison['confidence_comparison']]
                
                ax3.scatter(pytorch_conf, onnx_conf, alpha=0.6, color='green')
                ax3.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Match')
                ax3.set_xlabel('PyTorch Confidence')
                ax3.set_ylabel('ONNX Confidence')
                ax3.set_title('Confidence Score Comparison')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No matched detections\nfor confidence comparison', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Confidence Score Comparison')
            
            # 4. Detection match summary
            ax4 = axes[1, 1]
            match_types = ['Matched', 'PyTorch Only', 'ONNX Only']
            match_counts = [
                comparison['matched_detections'],
                comparison['pytorch_only'],
                comparison['onnx_only']
            ]
            colors = ['#2ca02c', '#d62728', '#9467bd']
            
            wedges, texts, autotexts = ax4.pie(match_counts, labels=match_types, colors=colors,
                                              autopct='%1.1f%%', startangle=90)
            ax4.set_title('Detection Match Summary')
            
            # Add count labels
            for i, (wedge, count) in enumerate(zip(wedges, match_counts)):
                if count > 0:
                    angle = (wedge.theta2 + wedge.theta1) / 2
                    x = 0.7 * np.cos(np.radians(angle))
                    y = 0.7 * np.sin(np.radians(angle))
                    ax4.text(x, y, f'({count})', ha='center', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Comparison visualization saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create comparison visualization: {e}")
            raise
    
    def create_performance_comparison(self, output_path: str) -> None:
        """
        Create performance comparison visualization.
        
        Args:
            output_path: Path to save the performance comparison
        """
        try:
            logger.info("Creating performance comparison...")
            
            # Performance metrics
            pytorch_time = self.pytorch_data['inference_time']
            onnx_time = self.onnx_data['inference_time']
            
            # Calculate speedup
            speedup = pytorch_time / onnx_time if onnx_time > 0 else 0
            
            # Create performance comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Performance Comparison: PyTorch vs ONNX', fontsize=14, fontweight='bold')
            
            # Inference time comparison
            models = ['PyTorch', 'ONNX']
            times = [pytorch_time, onnx_time]
            colors = ['#1f77b4', '#ff7f0e']
            
            bars = ax1.bar(models, times, color=colors, alpha=0.7)
            ax1.set_title('Inference Time Comparison')
            ax1.set_ylabel('Time (seconds)')
            
            # Add value labels
            for bar, time_val in zip(bars, times):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{time_val:.4f}s', ha='center', va='bottom', fontweight='bold')
            
            # Speedup visualization
            ax2.bar(['Speedup'], [speedup], color='#2ca02c', alpha=0.7)
            ax2.set_title(f'ONNX Speedup over PyTorch')
            ax2.set_ylabel('Speedup Factor')
            ax2.text(0, speedup + 0.05, f'{speedup:.2f}x', ha='center', va='bottom', 
                    fontweight='bold', fontsize=12)
            
            # Add horizontal line at 1.0 (no speedup)
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No Speedup')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Performance comparison saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create performance comparison: {e}")
            raise
    
    def save_comparison_report(self, comparison: Dict[str, Any], output_path: str) -> None:
        """
        Save detailed comparison report to file.
        
        Args:
            comparison: Comparison results dictionary
            output_path: Path to save the report
        """
        try:
            with open(output_path, 'w') as f:
                f.write("PyTorch vs ONNX Model Comparison Report\n")
                f.write("=" * 50 + "\n\n")
                
                # Executive Summary
                f.write("EXECUTIVE SUMMARY\n")
                f.write("-" * 20 + "\n")
                f.write(f"PyTorch Model Detections: {comparison['pytorch_detections']}\n")
                f.write(f"ONNX Model Detections: {comparison['onnx_detections']}\n")
                f.write(f"Matched Detections: {comparison['matched_detections']}\n")
                f.write(f"PyTorch Only: {comparison['pytorch_only']}\n")
                f.write(f"ONNX Only: {comparison['onnx_only']}\n")
                f.write(f"IoU Threshold: {comparison['iou_threshold']}\n\n")
                
                # Performance Comparison
                f.write("PERFORMANCE COMPARISON\n")
                f.write("-" * 25 + "\n")
                f.write(f"PyTorch Inference Time: {self.pytorch_data['inference_time']:.4f} seconds\n")
                f.write(f"ONNX Inference Time: {self.onnx_data['inference_time']:.4f} seconds\n")
                speedup = self.pytorch_data['inference_time'] / self.onnx_data['inference_time']
                f.write(f"ONNX Speedup: {speedup:.2f}x\n\n")
                
                # Class-wise Analysis
                f.write("CLASS-WISE ANALYSIS\n")
                f.write("-" * 20 + "\n")
                for class_name, data in comparison['class_comparison'].items():
                    f.write(f"{class_name}:\n")
                    f.write(f"  PyTorch: {data['pytorch']} detections\n")
                    f.write(f"  ONNX: {data['onnx']} detections\n")
                    f.write(f"  Difference: {data['difference']:+d}\n\n")
                
                # Confidence Analysis
                if comparison['confidence_comparison']:
                    f.write("CONFIDENCE SCORE ANALYSIS\n")
                    f.write("-" * 30 + "\n")
                    for i, comp in enumerate(comparison['confidence_comparison']):
                        f.write(f"Match {i+1} ({comp['class']}):\n")
                        f.write(f"  PyTorch Confidence: {comp['pytorch_confidence']:.4f}\n")
                        f.write(f"  ONNX Confidence: {comp['onnx_confidence']:.4f}\n")
                        f.write(f"  IoU: {comp['iou']:.4f}\n")
                        f.write(f"  Confidence Difference: {abs(comp['pytorch_confidence'] - comp['onnx_confidence']):.4f}\n\n")
                
                # Conclusions
                f.write("CONCLUSIONS\n")
                f.write("-" * 12 + "\n")
                if comparison['matched_detections'] > 0:
                    match_rate = comparison['matched_detections'] / max(comparison['pytorch_detections'], comparison['onnx_detections'])
                    f.write(f"• Match Rate: {match_rate:.2%} of detections have corresponding matches\n")
                
                if speedup > 1:
                    f.write(f"• ONNX model is {speedup:.2f}x faster than PyTorch model\n")
                else:
                    f.write(f"• PyTorch model is {1/speedup:.2f}x faster than ONNX model\n")
                
                if comparison['onnx_detections'] > comparison['pytorch_detections']:
                    f.write(f"• ONNX model detected {comparison['onnx_detections'] - comparison['pytorch_detections']} more objects\n")
                elif comparison['pytorch_detections'] > comparison['onnx_detections']:
                    f.write(f"• PyTorch model detected {comparison['pytorch_detections'] - comparison['onnx_detections']} more objects\n")
                else:
                    f.write("• Both models detected the same number of objects\n")
            
            logger.info(f"Comparison report saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save comparison report: {e}")
            raise


def main():
    """Main function to run results comparison."""
    # Configuration
    pytorch_results_file = "output/pytorch_results.txt"
    onnx_results_file = "output/onnx_results.txt"
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Output paths
    comparison_report = output_dir / "comparison_report.txt"
    comparison_viz = output_dir / "comparison_visualization.png"
    performance_viz = output_dir / "performance_comparison.png"
    
    try:
        logger.info("Starting PyTorch vs ONNX results comparison...")
        
        # Initialize comparator
        comparator = ResultsComparator(pytorch_results_file, onnx_results_file)
        
        # Compare detections
        comparison = comparator.compare_detections(iou_threshold=0.5)
        
        # Create visualizations
        comparator.create_comparison_visualization(comparison, comparison_viz)
        comparator.create_performance_comparison(performance_viz)
        
        # Save detailed report
        comparator.save_comparison_report(comparison, comparison_report)
        
        logger.info("Results comparison completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("PYTORCH vs ONNX COMPARISON SUMMARY")
        print("="*60)
        print(f"PyTorch Detections: {comparison['pytorch_detections']}")
        print(f"ONNX Detections: {comparison['onnx_detections']}")
        print(f"Matched Detections: {comparison['matched_detections']}")
        print(f"PyTorch Only: {comparison['pytorch_only']}")
        print(f"ONNX Only: {comparison['onnx_only']}")
        
        speedup = comparator.pytorch_data['inference_time'] / comparator.onnx_data['inference_time']
        print(f"ONNX Speedup: {speedup:.2f}x")
        
        print(f"\nReports saved to:")
        print(f"  - {comparison_report}")
        print(f"  - {comparison_viz}")
        print(f"  - {performance_viz}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Results comparison failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
