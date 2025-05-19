#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full ROI Pipeline for Dendrite and Spine Detection

This script runs the complete pipeline for automatic ROI selection for dendrites and spines.
It integrates data preparation, CNN training, and ROI detection.

Created on Sat May 18 15:30:00 2025

@author: minhnhitran
"""

import os
import sys
import time
import argparse
from typing import Dict, Any

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from run_roi_pipeline import run_pipeline as run_roi_pipeline
from run_roi_cnn import train_dendrite_cnn, train_spine_cnn, detect_rois


def measure_time(func):
    """Decorator to measure execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result, end_time - start_time
    return wrapper


@measure_time
def run_roi_preparation(
    tiff_dir: str,
    roi_file_path: str,
    output_dir: str,
    start_idx: int = 23,
    end_idx: int = 45
) -> Dict[str, Any]:
    """
    Run the ROI preparation pipeline.
    
    Args:
        tiff_dir: Directory containing TIFF files
        roi_file_path: Path to ROI file
        output_dir: Directory to save output
        start_idx: Start index of files to process
        end_idx: End index of files to process
        
    Returns:
        Dictionary containing pipeline results
    """
    # Create output directory
    roi_output_dir = os.path.join(output_dir, 'roi_pipeline')
    os.makedirs(roi_output_dir, exist_ok=True)
    
    # Run ROI pipeline
    run_roi_pipeline(
        tiff_dir,
        roi_file_path,
        roi_output_dir,
        start_idx,
        end_idx
    )
    
    # Return paths to training data
    return {
        'dendrite_data_dir': os.path.join(roi_output_dir, 'training_data', 'dendrite'),
        'spine_data_dir': os.path.join(roi_output_dir, 'training_data', 'spine')
    }


@measure_time
def run_dendrite_cnn_training(
    data_dir: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 16
) -> Dict[str, Any]:
    """
    Run dendrite CNN training.
    
    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save output
        epochs: Number of epochs to train for
        batch_size: Batch size
        
    Returns:
        Dictionary containing training results
    """
    # Create output directory
    dendrite_output_dir = os.path.join(output_dir, 'dendrite_cnn')
    os.makedirs(dendrite_output_dir, exist_ok=True)
    
    # Train dendrite CNN
    train_dendrite_cnn(
        data_dir,
        dendrite_output_dir,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Return path to trained model
    return {
        'model_path': os.path.join(dendrite_output_dir, 'checkpoints', 'best_model.pth')
    }


@measure_time
def run_spine_cnn_training(
    data_dir: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 16
) -> Dict[str, Any]:
    """
    Run spine CNN training.
    
    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save output
        epochs: Number of epochs to train for
        batch_size: Batch size
        
    Returns:
        Dictionary containing training results
    """
    # Create output directory
    spine_output_dir = os.path.join(output_dir, 'spine_cnn')
    os.makedirs(spine_output_dir, exist_ok=True)
    
    # Train spine CNN
    train_spine_cnn(
        data_dir,
        spine_output_dir,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Return path to trained model
    return {
        'model_path': os.path.join(spine_output_dir, 'checkpoints', 'best_model.pth')
    }


@measure_time
def run_roi_detection(
    tiff_dir: str,
    dendrite_model_path: str,
    spine_model_path: str,
    output_dir: str,
    start_idx: int = 23,
    end_idx: int = 45
) -> Dict[str, Any]:
    """
    Run ROI detection on test images.
    
    Args:
        tiff_dir: Directory containing TIFF files
        dendrite_model_path: Path to dendrite model
        spine_model_path: Path to spine model
        output_dir: Directory to save output
        start_idx: Start index of files to process
        end_idx: End index of files to process
        
    Returns:
        Dictionary containing detection results
    """
    # Create output directory
    detection_output_dir = os.path.join(output_dir, 'detection')
    os.makedirs(detection_output_dir, exist_ok=True)
    
    # Run ROI detection
    detect_rois(
        tiff_dir,
        dendrite_model_path,
        spine_model_path,
        detection_output_dir,
        start_idx=start_idx,
        end_idx=end_idx
    )
    
    # Return path to detection results
    return {
        'detection_dir': detection_output_dir
    }


def run_full_pipeline(
    tiff_dir: str,
    roi_file_path: str,
    output_dir: str,
    start_idx: int = 23,
    end_idx: int = 27,
    skip_training: bool = False,
    epochs: int = 50,
    batch_size: int = 16
) -> None:
    """
    Run the complete ROI detection pipeline.
    
    Args:
        tiff_dir: Directory containing TIFF files
        roi_file_path: Path to ROI file
        output_dir: Directory to save output
        start_idx: Start index of files to process
        end_idx: End index of files to process
        skip_training: Whether to skip model training
        epochs: Number of epochs for training
        batch_size: Batch size for training
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Record start time
    total_start_time = time.time()
    
    # Step 1: Run ROI preparation
    print("\n=== Step 1: ROI Preparation ===")
    (roi_data, roi_time) = run_roi_preparation(
        tiff_dir,
        roi_file_path,
        output_dir,
        start_idx,
        end_idx
    )
    
    # If skip_training is True, use pre-trained models
    if skip_training:
        print("\nSkipping model training, using pre-trained models...")
        dendrite_model_path = os.path.join(output_dir, 'dendrite_cnn', 'checkpoints', 'best_model.pth')
        spine_model_path = os.path.join(output_dir, 'spine_cnn', 'checkpoints', 'best_model.pth')
        
        # Check if pre-trained models exist
        if not os.path.exists(dendrite_model_path) or not os.path.exists(spine_model_path):
            print("Error: Pre-trained models not found. Please run with training enabled.")
            return
        
        dendrite_time = 0
        spine_time = 0
    else:
        # Step 2: Train dendrite CNN
        print("\n=== Step 2: Dendrite CNN Training ===")
        (dendrite_data, dendrite_time) = run_dendrite_cnn_training(
            roi_data['dendrite_data_dir'],
            output_dir,
            epochs=epochs,
            batch_size=batch_size
        )
        dendrite_model_path = dendrite_data['model_path']
        
        # Step 3: Train spine CNN
        print("\n=== Step 3: Spine CNN Training ===")
        (spine_data, spine_time) = run_spine_cnn_training(
            roi_data['spine_data_dir'],
            output_dir,
            epochs=epochs,
            batch_size=batch_size
        )
        spine_model_path = spine_data['model_path']
    
    # Step 4: Run ROI detection
    print("\n=== Step 4: ROI Detection ===")
    (detection_data, detection_time) = run_roi_detection(
        tiff_dir,
        dendrite_model_path,
        spine_model_path,
        output_dir,
        start_idx=start_idx,
        end_idx=end_idx
    )
    
    # Calculate total time
    total_time = time.time() - total_start_time
    
    # Generate performance report
    report_path = os.path.join(output_dir, 'full_pipeline_report.txt')
    with open(report_path, 'w') as f:
        f.write("Full Pipeline Report\n")
        f.write("===================\n\n")
        f.write(f"TIFF Files: {end_idx - start_idx} processed\n")
        f.write(f"ROI File: {os.path.basename(roi_file_path)}\n\n")
        
        f.write("Execution Times:\n")
        f.write(f"  ROI Preparation: {roi_time:.2f} seconds\n")
        
        if not skip_training:
            f.write(f"  Dendrite CNN Training: {dendrite_time:.2f} seconds\n")
            f.write(f"  Spine CNN Training: {spine_time:.2f} seconds\n")
        else:
            f.write("  Model Training: Skipped\n")
        
        f.write(f"  ROI Detection: {detection_time:.2f} seconds\n")
        f.write(f"  Total: {total_time:.2f} seconds\n")
    
    print(f"\nFull pipeline completed in {total_time:.2f} seconds")
    print(f"Results saved to: {output_dir}")
    print(f"Performance report: {report_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Full ROI Pipeline for Dendrite and Spine Detection')
    parser.add_argument('--tiff-dir', type=str, default='/Volumes/TOSHIBA EXT/MT033/1/tiff',
                        help='Directory containing TIFF files')
    parser.add_argument('--roi-file', type=str, 
                        default='/Volumes/TOSHIBA EXT/MT033/1/rois/MT033_01_2023-11-22_00001_00011_Turboreg.roi',
                        help='Path to ROI file')
    parser.add_argument('--output-dir', type=str, default='/Users/minhnhitran/Desktop/Github/Minh_ROI_semi_autodetection/outputs',
                        help='Directory to save output')
    parser.add_argument('--start-idx', type=int, default=23,
                        help='Start index of files to process')
    parser.add_argument('--end-idx', type=int, default=27,
                        help='End index of files to process')
    parser.add_argument('--no-train', action='store_true',
                        help='Skip model training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    
    args = parser.parse_args()
    
    # Run full pipeline
    run_full_pipeline(
        args.tiff_dir,
        args.roi_file,
        args.output_dir,
        args.start_idx,
        args.end_idx,
        skip_training=args.no_train,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
