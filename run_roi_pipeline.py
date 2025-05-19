#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI Pipeline for Dendrite and Spine Detection

This script implements the automatic ROI selection pipeline for dendrites and spines.
It processes TIFF files and extracts ROIs for dendrites and spines.

Created on Sat May 18 15:04:00 2025

@author: minhnhitran
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from typing import List, Tuple, Dict, Union, Optional, Any
import argparse

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.utils.tiff_handling import (
    load_tiff_stack, create_max_projection, preprocess_frame,
    extract_frames_from_stack, split_image_into_patches,
    filter_patches_with_dendrites
)
from src.utils.roi_processing import (
    load_roi_file, convert_polygons_to_masks, classify_roi_type,
    associate_spines_with_dendrites, calculate_roi_properties
)
from src.utils.visualization import (
    visualize_roi_polygons, visualize_roi_masks,
    visualize_spine_dendrite_associations, visualize_roi_properties,
    visualize_roi_classification
)
from src.utils.data_augmentation import (
    augment_image_and_mask, generate_augmented_dataset
)


def measure_time(func):
    """Decorator to measure execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result, end_time - start_time
    return wrapper


def find_tiff_files(tiff_dir: str) -> List[str]:
    """
    Find TIFF files in the specified directory.
    
    Args:
        tiff_dir: Directory containing TIFF files
        
    Returns:
        List of TIFF file paths
    """
    # Find all TIFF files
    tiff_files = []
    for ext in ['.tif', '.tiff']:
        tiff_files.extend(glob(os.path.join(tiff_dir, f'*{ext}')))
    
    # Sort files
    tiff_files = sorted(tiff_files)
    
    return tiff_files


@measure_time
def load_and_preprocess_tiff_files(tiff_files: List[str], start_idx: int, end_idx: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load and preprocess TIFF files.
    
    Args:
        tiff_files: List of TIFF file paths
        start_idx: Start index of files to process
        end_idx: End index of files to process
        
    Returns:
        Tuple of (raw_frames, preprocessed_frames)
    """
    # Select files to process
    selected_files = tiff_files[start_idx:end_idx]
    
    # Lists to store frames
    raw_frames = []
    preprocessed_frames = []
    
    # Process each file
    for file_path in selected_files:
        print(f"Processing {os.path.basename(file_path)}")
        
        # Load TIFF stack
        stack = load_tiff_stack(file_path)
        
        # Create maximum intensity projection
        max_proj = create_max_projection(stack)
        
        # Preprocess frame
        preprocessed = preprocess_frame(max_proj)
        
        # Add to lists
        raw_frames.append(max_proj)
        preprocessed_frames.append(preprocessed)
    
    return raw_frames, preprocessed_frames


@measure_time
def load_and_process_roi_file(roi_file_path: str, image_shape: Tuple[int, int]) -> Dict[str, Any]:
    """
    Load and process ROI file.
    
    Args:
        roi_file_path: Path to ROI file
        image_shape: Shape of the image (height, width)
        
    Returns:
        Dictionary containing processed ROI data
    """
    # Load ROI file
    roi_data = load_roi_file(roi_file_path)
    
    # Extract polygons
    polygons = roi_data.get('polygons', [])
    
    if not polygons:
        raise ValueError(f"No polygons found in ROI file: {roi_file_path}")
    
    # Convert polygons to masks
    masks = convert_polygons_to_masks(polygons, image_shape)
    
    # Classify ROIs
    classifications = []
    properties = []
    
    for mask in masks:
        # Calculate properties
        props = calculate_roi_properties(mask)
        properties.append(props)
        
        # Classify ROI
        roi_type = classify_roi_type(mask)
        classifications.append(roi_type)
    
    # Separate dendrites and spines
    dendrite_masks = [mask for mask, cls in zip(masks, classifications) if cls == 'dendrite']
    spine_masks = [mask for mask, cls in zip(masks, classifications) if cls == 'spine']
    
    dendrite_polygons = [poly for poly, cls in zip(polygons, classifications) if cls == 'dendrite']
    spine_polygons = [poly for poly, cls in zip(polygons, classifications) if cls == 'spine']
    
    # Associate spines with dendrites
    associations = associate_spines_with_dendrites(spine_masks, dendrite_masks)
    
    # Create combined masks
    combined_dendrite_mask = np.zeros(image_shape, dtype=np.uint8)
    for mask in dendrite_masks:
        combined_dendrite_mask = np.logical_or(combined_dendrite_mask, mask).astype(np.uint8)
    
    combined_spine_mask = np.zeros(image_shape, dtype=np.uint8)
    for mask in spine_masks:
        combined_spine_mask = np.logical_or(combined_spine_mask, mask).astype(np.uint8)
    
    # Return processed data
    return {
        'polygons': polygons,
        'masks': masks,
        'classifications': classifications,
        'properties': properties,
        'dendrite_masks': dendrite_masks,
        'spine_masks': spine_masks,
        'dendrite_polygons': dendrite_polygons,
        'spine_polygons': spine_polygons,
        'associations': associations,
        'combined_dendrite_mask': combined_dendrite_mask,
        'combined_spine_mask': combined_spine_mask
    }


@measure_time
def prepare_training_data(
    images: List[np.ndarray],
    roi_data: Dict[str, Any],
    output_dir: str,
    n_augmentations: int = 5
) -> Dict[str, Any]:
    """
    Prepare training data for dendrite and spine detection.
    
    Args:
        images: List of preprocessed images
        roi_data: Dictionary containing ROI data
        output_dir: Directory to save training data
        n_augmentations: Number of augmentations per image
        
    Returns:
        Dictionary containing training data information
    """
    # Create output directories
    dendrite_dir = os.path.join(output_dir, 'dendrite')
    spine_dir = os.path.join(output_dir, 'spine')
    
    os.makedirs(dendrite_dir, exist_ok=True)
    os.makedirs(spine_dir, exist_ok=True)
    
    # Extract masks
    dendrite_masks = roi_data['dendrite_masks']
    spine_masks = roi_data['spine_masks']
    combined_dendrite_mask = roi_data['combined_dendrite_mask']
    
    # Prepare dendrite training data
    print("Preparing dendrite training data...")
    
    # Create augmented dataset
    augmented_images, augmented_dendrite_masks = generate_augmented_dataset(
        images, [combined_dendrite_mask] * len(images), n_augmentations
    )
    
    # Save augmented dataset
    for i, (img, mask) in enumerate(zip(augmented_images, augmented_dendrite_masks)):
        # Save image
        img_path = os.path.join(dendrite_dir, f'image_{i:04d}.npy')
        np.save(img_path, img)
        
        # Save mask
        mask_path = os.path.join(dendrite_dir, f'mask_{i:04d}.npy')
        np.save(mask_path, mask)
    
    # Prepare spine training data
    print("Preparing spine training data...")
    
    # Extract patches containing dendrites
    all_patches = []
    all_patch_masks = []
    all_coordinates = []
    
    for img in images:
        # Split image into patches
        patches, coordinates = split_image_into_patches(img)
        
        # Filter patches containing dendrites
        filtered_patches, filtered_coordinates = filter_patches_with_dendrites(
            patches, coordinates, combined_dendrite_mask, threshold=0.05
        )
        
        # Extract corresponding spine masks
        patch_masks = []
        for y, x in filtered_coordinates:
            # Extract patch from spine mask
            mask_patch = combined_dendrite_mask[y:y+128, x:x+128]
            patch_masks.append(mask_patch)
        
        # Add to lists
        all_patches.extend(filtered_patches)
        all_patch_masks.extend(patch_masks)
        all_coordinates.extend(filtered_coordinates)
    
    # Create augmented dataset
    augmented_patches, augmented_patch_masks = generate_augmented_dataset(
        all_patches, all_patch_masks, n_augmentations
    )
    
    # Save augmented dataset
    for i, (patch, mask) in enumerate(zip(augmented_patches, augmented_patch_masks)):
        # Save patch
        patch_path = os.path.join(spine_dir, f'patch_{i:04d}.npy')
        np.save(patch_path, patch)
        
        # Save mask
        mask_path = os.path.join(spine_dir, f'mask_{i:04d}.npy')
        np.save(mask_path, mask)
    
    # Return training data information
    return {
        'dendrite_data': {
            'images': len(augmented_images),
            'masks': len(augmented_dendrite_masks)
        },
        'spine_data': {
            'patches': len(augmented_patches),
            'masks': len(augmented_patch_masks)
        }
    }


@measure_time
def visualize_roi_data(
    image: np.ndarray,
    roi_data: Dict[str, Any],
    output_dir: str
) -> None:
    """
    Visualize ROI data.
    
    Args:
        image: Image to visualize
        roi_data: Dictionary containing ROI data
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    polygons = roi_data['polygons']
    masks = roi_data['masks']
    classifications = roi_data['classifications']
    properties = roi_data['properties']
    dendrite_masks = roi_data['dendrite_masks']
    spine_masks = roi_data['spine_masks']
    associations = roi_data['associations']
    
    # Convert classifications to numeric
    roi_types = [0 if cls == 'dendrite' else 1 for cls in classifications]
    
    # Visualize ROI polygons
    visualize_roi_polygons(
        image, polygons, roi_types,
        save_path=os.path.join(output_dir, 'roi_polygons.png')
    )
    
    # Visualize ROI masks
    visualize_roi_masks(
        image, masks,
        save_path=os.path.join(output_dir, 'roi_masks.png')
    )
    
    # Visualize ROI classification
    visualize_roi_classification(
        image, masks, classifications,
        save_path=os.path.join(output_dir, 'roi_classification.png')
    )
    
    # Visualize spine-dendrite associations
    visualize_spine_dendrite_associations(
        image, dendrite_masks, spine_masks, associations,
        save_path=os.path.join(output_dir, 'spine_dendrite_associations.png')
    )
    
    # Visualize ROI properties
    visualize_roi_properties(
        properties,
        save_path=os.path.join(output_dir, 'roi_properties.png')
    )


def run_pipeline(
    tiff_dir: str,
    roi_file_path: str,
    output_dir: str,
    start_idx: int = 23,
    end_idx: int = 45
) -> None:
    """
    Run the complete ROI detection pipeline.
    
    Args:
        tiff_dir: Directory containing TIFF files
        roi_file_path: Path to ROI file
        output_dir: Directory to save output
        start_idx: Start index of files to process
        end_idx: End index of files to process
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Record start time
    total_start_time = time.time()
    
    # Step 1: Find TIFF files
    print("Finding TIFF files...")
    tiff_files = find_tiff_files(tiff_dir)
    print(f"Found {len(tiff_files)} TIFF files")
    
    # Step 2: Load and preprocess TIFF files
    print(f"\nProcessing TIFF files {start_idx} to {end_idx}...")
    (raw_frames, preprocessed_frames), preprocessing_time = load_and_preprocess_tiff_files(
        tiff_files, start_idx, end_idx
    )
    
    # Step 3: Load and process ROI file
    print("\nProcessing ROI file...")
    image_shape = raw_frames[0].shape if raw_frames else (512, 512)
    roi_data, roi_processing_time = load_and_process_roi_file(roi_file_path, image_shape)
    
    # Step 4: Visualize ROI data
    print("\nVisualizing ROI data...")
    visualize_dir = os.path.join(output_dir, 'visualizations')
    visualize_roi_data(raw_frames[0], roi_data, visualize_dir)
    
    # Step 5: Prepare training data
    print("\nPreparing training data...")
    training_dir = os.path.join(output_dir, 'training_data')
    training_data, training_time = prepare_training_data(
        preprocessed_frames, roi_data, training_dir
    )
    
    # Calculate total time
    total_time = time.time() - total_start_time
    
    # Generate performance report
    report_path = os.path.join(output_dir, 'performance_report.txt')
    with open(report_path, 'w') as f:
        f.write("Performance Report\n")
        f.write("=================\n\n")
        f.write(f"TIFF Files: {len(tiff_files)} total, {end_idx - start_idx} processed\n")
        f.write(f"ROI File: {os.path.basename(roi_file_path)}\n\n")
        
        f.write("ROI Statistics:\n")
        f.write(f"  Total ROIs: {len(roi_data['polygons'])}\n")
        f.write(f"  Dendrites: {len(roi_data['dendrite_masks'])}\n")
        f.write(f"  Spines: {len(roi_data['spine_masks'])}\n\n")
        
        f.write("Training Data:\n")
        f.write(f"  Dendrite Images: {training_data['dendrite_data']['images']}\n")
        f.write(f"  Spine Patches: {training_data['spine_data']['patches']}\n\n")
        
        f.write("Execution Times:\n")
        f.write(f"  Preprocessing: {preprocessing_time:.2f} seconds\n")
        f.write(f"  ROI Processing: {roi_processing_time:.2f} seconds\n")
        f.write(f"  Training Data Preparation: {training_time:.2f} seconds\n")
        f.write(f"  Total: {total_time:.2f} seconds\n")
    
    print(f"\nPipeline completed in {total_time:.2f} seconds")
    print(f"Results saved to: {output_dir}")
    print(f"Performance report: {report_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='ROI Pipeline for Dendrite and Spine Detection')
    parser.add_argument('--tiff-dir', type=str, default='/Volumes/TOSHIBA EXT/MT033/1/tiff',
                        help='Directory containing TIFF files')
    parser.add_argument('--roi-file', type=str, 
                        default='/Volumes/TOSHIBA EXT/MT033/1/rois/MT033_01_2023-11-22_00001_00011_Turboreg.roi',
                        help='Path to ROI file')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Directory to save output')
    parser.add_argument('--start-idx', type=int, default=23,
                        help='Start index of files to process')
    parser.add_argument('--end-idx', type=int, default=45,
                        help='End index of files to process')
    
    args = parser.parse_args()
    
    # Run pipeline
    run_pipeline(
        args.tiff_dir,
        args.roi_file,
        args.output_dir,
        args.start_idx,
        args.end_idx
    )


if __name__ == "__main__":
    main()
