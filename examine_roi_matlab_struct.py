#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 11:01:39 2025

@author: minhnhitran
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pprint

def examine_polygon_structure(file_path):
    """
    Examine the structure of the polygon variable in a MATLAB .roi file.
    
    Args:
        file_path (str): Path to the MATLAB .roi file
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return
    
    # Load the MATLAB file
    try:
        mat_data = loadmat(file_path)
        print(f"Successfully loaded {file_path}")
        
        # Check if 'polygon' variable exists
        if 'polygon' not in mat_data:
            print("Error: 'polygon' variable not found in the file")
            print(f"Available variables: {list(mat_data.keys())}")
            return
        
        # Get the polygon variable
        polygon = mat_data['polygon']
        
        # Print basic information
        print("\n=== Basic Information ===")
        print(f"Type: {type(polygon)}")
        print(f"Shape: {polygon.shape}")
        print(f"Data type: {polygon.dtype}")
        
        # Examine the structure more deeply
        print("\n=== Structure Analysis ===")
        
        # If it's a structured array, examine its fields
        if polygon.dtype.names is not None:
            print("This is a structured array with fields:")
            for field in polygon.dtype.names:
                print(f"  - {field}: {polygon[field].shape}")
                
            # Print sample of each field
            print("\n=== Field Samples ===")
            for field in polygon.dtype.names:
                print(f"Field: {field}")
                try:
                    sample = polygon[field][0]
                    if isinstance(sample, np.ndarray) and sample.size > 10:
                        print(f"  Sample (first 10 elements): {sample.flatten()[:10]}")
                    else:
                        print(f"  Sample: {sample}")
                except:
                    print("  Could not extract sample")
        
        # If it's a cell array (common in MATLAB)
        elif polygon.shape == (1, 1) and isinstance(polygon[0, 0], np.ndarray):
            print("This appears to be a MATLAB cell array")
            cell_content = polygon[0, 0]
            print(f"Cell content type: {type(cell_content)}")
            print(f"Cell content shape: {cell_content.shape}")
            
            # If the cell contains coordinates
            if cell_content.ndim == 2 and cell_content.shape[1] == 2:
                print("This appears to be a list of (x,y) coordinates")
                print(f"Number of coordinates: {cell_content.shape[0]}")
                print(f"First 5 coordinates: {cell_content[:5]}")
                
                # Visualize the ROI
                plt.figure(figsize=(10, 10))
                plt.plot(cell_content[:, 0], cell_content[:, 1], 'b-')
                plt.scatter(cell_content[:, 0], cell_content[:, 1], c='r', s=10)
                plt.title("ROI Coordinates")
                plt.axis('equal')
                plt.grid(True)
                plt.show()
            else:
                print(f"Cell content (first 10 elements): {cell_content.flatten()[:10]}")
        
        # If it's a nested array structure
        elif polygon.ndim > 1:
            print("This appears to be a nested array structure")
            
            # Try to navigate the structure
            def explore_nested(arr, depth=0, max_depth=3):
                if depth >= max_depth:
                    return
                
                if isinstance(arr, np.ndarray):
                    print(f"{'  ' * depth}Array shape: {arr.shape}, type: {arr.dtype}")
                    
                    if arr.size > 0:
                        if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] <= 10):
                            print(f"{'  ' * depth}Content: {arr}")
                        else:
                            # For larger arrays, just show the first element
                            if arr.size > 0:
                                first_item = arr.item(0) if arr.size == 1 else arr[0]
                                print(f"{'  ' * depth}First element: {first_item}")
                                explore_nested(first_item, depth + 1, max_depth)
            
            explore_nested(polygon)
        
        # If it's a simple array
        else:
            print("This appears to be a simple array")
            print(f"Content: {polygon}")
        
        # Additional analysis for any type
        print("\n=== Additional Analysis ===")
        
        # Check for NaN or Inf values
        if np.isnan(polygon).any():
            print("Warning: Contains NaN values")
        
        if np.isinf(polygon).any():
            print("Warning: Contains Inf values")
        
        # Try to determine if it's a closed polygon
        try:
            if isinstance(polygon[0, 0], np.ndarray):
                coords = polygon[0, 0]
                if coords.shape[1] == 2:
                    if np.array_equal(coords[0], coords[-1]):
                        print("This is a closed polygon (first point equals last point)")
                    else:
                        print("This is an open polygon (first point differs from last point)")
        except:
            pass
        
        print("\n=== Raw Data Structure ===")
        # Use pretty printer for better readability
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(polygon.tolist())
        
        return polygon
    
    except Exception as e:
        print(f"Error examining file: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to examine a MATLAB .roi file."""
    # Get file path from user
    file_path = input("Enter the path to the MATLAB .roi file: ")
    
    # Examine the file
    examine_polygon_structure(file_path)

if __name__ == '__main__':
    main()
