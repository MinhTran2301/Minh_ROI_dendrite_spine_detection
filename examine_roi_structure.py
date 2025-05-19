#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to examine the structure of ROI data in MATLAB .roi files.

This script specifically checks for structured arrays with a 'ROI' field
and attempts to extract polygon coordinates from them.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Global variables that will be visible in Variable Explorer
data = None
polygon_data = None
roi_data = None
roi_item = None
polygons = []

def examine_roi_structure(file_path):
    """
    Examine the structure of a .roi file, specifically looking for 'ROI' fields.
    
    Args:
        file_path (str): Path to the .roi file
    """
    global data, polygon_data, roi_data, roi_item, polygons
    
    print(f"Examining file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return
    
    # Load the MATLAB file
    try:
        data = loadmat(file_path)
        print(f"Successfully loaded {file_path}")
        
        # Remove MATLAB-specific keys
        for key in list(data.keys()):
            if key.startswith('__'):
                del data[key]
        
        print(f"Available variables: {list(data.keys())}")
        
        # Check if 'polygon' variable exists
        if 'polygon' not in data:
            print("Error: 'polygon' variable not found in the file")
            return
        
        # Get the polygon variable
        polygon_data = data['polygon']
        
        # Print basic information
        print("\n=== Basic Information ===")
        print(f"Type: {type(polygon_data)}")
        print(f"Shape: {polygon_data.shape}")
        print(f"Data type: {polygon_data.dtype}")
        
        # Check if it's a structured array
        if polygon_data.dtype.names is not None:
            print("\n=== Structured Array Fields ===")
            print(f"Field names: {polygon_data.dtype.names}")
            
            # Check if 'ROI' field exists
            if 'ROI' in polygon_data.dtype.names:
                print("\n=== Found 'ROI' field! ===")
                roi_data = polygon_data['ROI']
                print(f"ROI data type: {type(roi_data)}")
                print(f"ROI data shape: {roi_data.shape}")
                
                # Try to extract polygon coordinates
                try:
                    # If it's a single item
                    if roi_data.shape == (1, 1):
                        roi_item = roi_data[0, 0]
                        print(f"ROI item type: {type(roi_item)}")
                        
                        # If it's an array of polygons
                        if isinstance(roi_item, np.ndarray):
                            print(f"ROI item shape: {roi_item.shape}")
                            print(f"ROI item dtype: {roi_item.dtype}")
                            
                            # Clear previous polygons
                            polygons = []
                            
                            # If it's a 1D array of polygons
                            if roi_item.ndim == 1:
                                print("\n=== Extracting polygons from 1D array ===")
                                for i, poly in enumerate(roi_item):
                                    if isinstance(poly, np.ndarray) and poly.ndim == 2 and poly.shape[1] == 2:
                                        polygons.append(poly)
                                        print(f"Found polygon {i+1} with shape {poly.shape}")
                                        # Print first few coordinates
                                        print(f"First 3 coordinates: {poly[:3]}")
                            
                            # If it's a 2D array with shape (N, 2), it's a single polygon
                            elif roi_item.ndim == 2 and roi_item.shape[1] == 2:
                                print("\n=== Found a single polygon ===")
                                polygons.append(roi_item)
                                print(f"Polygon shape: {roi_item.shape}")
                                print(f"First 3 coordinates: {roi_item[:3]}")
                            
                            # Visualize polygons if found
                            if polygons:
                                print(f"\nFound {len(polygons)} polygons!")
                                
                                # Create individual variables for each polygon for Variable Explorer
                                for i, poly in enumerate(polygons):
                                    # Use exec to create variables dynamically
                                    exec(f"global polygon_{i+1}; polygon_{i+1} = poly", globals(), locals())
                                    print(f"Created variable 'polygon_{i+1}' with shape {poly.shape}")
                                
                                # Plot polygons
                                plt.figure(figsize=(10, 10))
                                for i, poly in enumerate(polygons):
                                    plt.plot(poly[:, 0], poly[:, 1], '-', label=f'Polygon {i+1}')
                                    plt.scatter(poly[:, 0], poly[:, 1], s=10)
                                
                                plt.title("ROI Polygons")
                                plt.legend()
                                plt.axis('equal')
                                plt.grid(True)
                                plt.show()
                            else:
                                print("No valid polygons found in ROI field")
                        else:
                            print("ROI item is not a numpy array")
                    else:
                        print("ROI data has unexpected shape")
                
                except Exception as e:
                    print(f"Error extracting polygons: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                print("'ROI' field not found in structured array")
                print(f"Available fields: {polygon_data.dtype.names}")
                
                # Try each field to see if it contains polygon data
                for field_name in polygon_data.dtype.names:
                    print(f"\nTrying field: {field_name}")
                    field_data = polygon_data[field_name]
                    print(f"Field data type: {type(field_data)}")
                    print(f"Field data shape: {field_data.shape}")
                    
                    # Make this field data available in Variable Explorer
                    exec(f"global field_{field_name}; field_{field_name} = field_data", globals(), locals())
                    print(f"Created variable 'field_{field_name}' for Variable Explorer")
        else:
            print("Polygon data is not a structured array")
            
            # Try to navigate the structure
            print("\n=== Attempting to navigate the structure ===")
            
            try:
                # If it's a 1D array of polygons
                if polygon_data.ndim == 1:
                    print("Polygon data is a 1D array")
                    for i, item in enumerate(polygon_data):
                        print(f"Item {i} type: {type(item)}")
                        if hasattr(item, 'shape'):
                            print(f"Item {i} shape: {item.shape}")
                        
                        # Make this item available in Variable Explorer
                        exec(f"global item_{i}; item_{i} = item", globals(), locals())
                        print(f"Created variable 'item_{i}' for Variable Explorer")
                
                # If it's a 2D array
                elif polygon_data.ndim == 2:
                    print("Polygon data is a 2D array")
                    if polygon_data.shape[1] == 2:
                        print("This appears to be a single polygon")
                        # Make it available in Variable Explorer
                        global single_polygon
                        single_polygon = polygon_data
                        print("Created variable 'single_polygon' for Variable Explorer")
                    else:
                        print(f"Shape: {polygon_data.shape}")
                        
                        # Check first item
                        if polygon_data.size > 0:
                            first_item = polygon_data[0, 0]
                            print(f"First item type: {type(first_item)}")
                            if hasattr(first_item, 'shape'):
                                print(f"First item shape: {first_item.shape}")
                            
                            # Make first item available in Variable Explorer
                            global first_array_item
                            first_array_item = first_item
                            print("Created variable 'first_array_item' for Variable Explorer")
                            
                            # If it's a numpy array, try to extract it
                            if isinstance(first_item, np.ndarray):
                                print("\n=== First item content ===")
                                print(first_item)
            
            except Exception as e:
                print(f"Error navigating structure: {str(e)}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"Error examining file: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Return the important variables to make them available in the global scope
    return data, polygon_data, roi_data, roi_item, polygons

def main():
    """Main function to examine a .roi file."""
    # Get file path from user
    file_path = input("Enter the path to the .roi file: ")
    
    # Examine the file
    examine_roi_structure(file_path)
    
    # Print a message to remind the user to check Variable Explorer
    print("\n=== Check Variable Explorer for extracted arrays ===")
    print("The following variables should be available:")
    print("- data: The full MATLAB file data")
    print("- polygon_data: The polygon variable from the file")
    print("- roi_data: The ROI field data if found")
    print("- roi_item: The extracted ROI item if found")
    print("- polygons: List of extracted polygon arrays")
    print("- polygon_1, polygon_2, etc.: Individual polygon arrays")

# Run the script if executed directly
if __name__ == '__main__':
    main()
