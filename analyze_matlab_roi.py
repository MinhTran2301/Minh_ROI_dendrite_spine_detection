"""
Script to analyze the structure of MATLAB .roi files.

This script reads a MATLAB .roi file and attempts to determine its structure
by analyzing the binary content. It will print information about the file
format and any detected patterns.
"""

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import io

def analyze_binary_file(file_path):
    """
    Analyze a binary file to determine its structure.
    
    Args:
        file_path (str): Path to the binary file
        
    Returns:
        dict: Information about the file structure
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return None
    
    # Get file size
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size} bytes")
    
    # Read the entire file as binary
    with open(file_path, 'rb') as f:
        data = f.read()
    
    # Try to determine if it's a MATLAB file
    if data[:2] == b'MA' or data[:4] == b'MATL':
        print("This appears to be a MATLAB .mat file")
        try:
            # Try to load as a .mat file
            mat_data = loadmat(file_path)
            print("Successfully loaded as MATLAB .mat file")
            print("Contents:")
            for key in mat_data.keys():
                if not key.startswith('__'):  # Skip metadata
                    print(f"  {key}: {type(mat_data[key])}, shape: {mat_data[key].shape}")
            return {'type': 'mat', 'data': mat_data}
        except Exception as e:
            print(f"Error loading as .mat file: {str(e)}")
    
    # Check for common binary patterns
    print("\nAnalyzing binary patterns...")
    
    # Look for potential header information
    print("\nFirst 64 bytes (potential header):")
    for i in range(0, min(64, len(data)), 4):
        hex_values = ' '.join([f"{b:02X}" for b in data[i:i+4]])
        ascii_values = ''.join([chr(b) if 32 <= b <= 126 else '.' for b in data[i:i+4]])
        print(f"Offset {i:04X}: {hex_values}  {ascii_values}")
    
    # Try to interpret as different data types
    print("\nTrying to interpret data as different types...")
    
    # Try to interpret as int32 array
    if len(data) >= 4:
        try:
            int32_values = []
            for i in range(0, min(40, len(data)), 4):
                val = struct.unpack('>i', data[i:i+4])[0]  # Big-endian
                int32_values.append(val)
            print(f"As int32 (big-endian): {int32_values}")
            
            int32_values = []
            for i in range(0, min(40, len(data)), 4):
                val = struct.unpack('<i', data[i:i+4])[0]  # Little-endian
                int32_values.append(val)
            print(f"As int32 (little-endian): {int32_values}")
        except Exception as e:
            print(f"Error interpreting as int32: {str(e)}")
    
    # Try to interpret as float32 array
    if len(data) >= 4:
        try:
            float32_values = []
            for i in range(0, min(40, len(data)), 4):
                val = struct.unpack('>f', data[i:i+4])[0]  # Big-endian
                float32_values.append(val)
            print(f"As float32 (big-endian): {float32_values}")
            
            float32_values = []
            for i in range(0, min(40, len(data)), 4):
                val = struct.unpack('<f', data[i:i+4])[0]  # Little-endian
                float32_values.append(val)
            print(f"As float32 (little-endian): {float32_values}")
        except Exception as e:
            print(f"Error interpreting as float32: {str(e)}")
    
    # Look for potential coordinate data (pairs of values)
    print("\nLooking for potential coordinate data...")
    
    # Try to interpret as pairs of int16 values (x,y coordinates)
    if len(data) >= 4:
        try:
            coords = []
            for i in range(0, min(100, len(data)), 4):
                x = struct.unpack('>h', data[i:i+2])[0]  # Big-endian
                y = struct.unpack('>h', data[i+2:i+4])[0]
                coords.append((x, y))
            print(f"As int16 coordinate pairs (big-endian): {coords[:10]}")
            
            coords = []
            for i in range(0, min(100, len(data)), 4):
                x = struct.unpack('<h', data[i:i+2])[0]  # Little-endian
                y = struct.unpack('<h', data[i+2:i+4])[0]
                coords.append((x, y))
            print(f"As int16 coordinate pairs (little-endian): {coords[:10]}")
        except Exception as e:
            print(f"Error interpreting as int16 coordinates: {str(e)}")
    
    # Try to interpret as pairs of float32 values (x,y coordinates)
    if len(data) >= 8:
        try:
            coords = []
            for i in range(0, min(100, len(data)), 8):
                if i + 8 <= len(data):
                    x = struct.unpack('>f', data[i:i+4])[0]  # Big-endian
                    y = struct.unpack('>f', data[i+4:i+8])[0]
                    coords.append((x, y))
            print(f"As float32 coordinate pairs (big-endian): {coords[:10]}")
            
            coords = []
            for i in range(0, min(100, len(data)), 8):
                if i + 8 <= len(data):
                    x = struct.unpack('<f', data[i:i+4])[0]  # Little-endian
                    y = struct.unpack('<f', data[i+4:i+8])[0]
                    coords.append((x, y))
            print(f"As float32 coordinate pairs (little-endian): {coords[:10]}")
        except Exception as e:
            print(f"Error interpreting as float32 coordinates: {str(e)}")
    
    # Try to find patterns that might indicate ROI data
    print("\nLooking for patterns that might indicate ROI data...")
    
    # Count occurrences of common values
    value_counts = {}
    for i in range(len(data)):
        b = data[i]
        if b in value_counts:
            value_counts[b] += 1
        else:
            value_counts[b] = 1
    
    # Print most common values
    print("Most common byte values:")
    sorted_counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
    for b, count in sorted_counts[:10]:
        print(f"  0x{b:02X} ({b}): {count} occurrences ({count/len(data)*100:.2f}%)")
    
    # Look for potential length indicators
    print("\nPotential length indicators:")
    for i in range(0, min(100, len(data)), 4):
        if i + 4 <= len(data):
            val_be = struct.unpack('>I', data[i:i+4])[0]  # Big-endian
            val_le = struct.unpack('<I', data[i:i+4])[0]  # Little-endian
            if 10 < val_be < 10000 and val_be < len(data):
                print(f"  Offset {i:04X}: {val_be} (big-endian)")
            if 10 < val_le < 10000 and val_le < len(data):
                print(f"  Offset {i:04X}: {val_le} (little-endian)")
    
    return {'type': 'unknown', 'data': data}

def try_visualize_as_roi(data, offset=0, length=None, step=2, as_int16=True, big_endian=True):
    """
    Try to visualize a section of binary data as ROI coordinates.
    
    Args:
        data (bytes): Binary data
        offset (int): Starting offset in the data
        length (int): Number of bytes to interpret (None for all remaining)
        step (int): Step size between coordinates (2 for int16, 4 for float32)
        as_int16 (bool): If True, interpret as int16, otherwise as float32
        big_endian (bool): If True, use big-endian, otherwise little-endian
        
    Returns:
        list: List of (x,y) coordinates
    """
    if length is None:
        length = len(data) - offset
    
    end = min(offset + length, len(data))
    
    coords = []
    format_char = '>' if big_endian else '<'
    format_char += 'h' if as_int16 else 'f'
    bytes_per_value = 2 if as_int16 else 4
    
    for i in range(offset, end, bytes_per_value * 2):
        if i + bytes_per_value * 2 <= end:
            try:
                x = struct.unpack(format_char, data[i:i+bytes_per_value])[0]
                y = struct.unpack(format_char, data[i+bytes_per_value:i+bytes_per_value*2])[0]
                coords.append((x, y))
            except Exception:
                break
    
    # Plot the coordinates
    if coords:
        plt.figure(figsize=(10, 10))
        coords_array = np.array(coords)
        plt.plot(coords_array[:, 0], coords_array[:, 1], 'b-')
        plt.scatter(coords_array[:, 0], coords_array[:, 1], c='r', s=10)
        
        # Add coordinate labels for some points
        for i in range(0, len(coords), max(1, len(coords) // 10)):
            plt.text(coords[i][0], coords[i][1], f"{i}", fontsize=8)
        
        plt.title(f"Potential ROI Coordinates (offset={offset}, {'int16' if as_int16 else 'float32'}, {'big' if big_endian else 'little'}-endian)")
        plt.axis('equal')
        plt.grid(True)
        plt.show()
    
    return coords

def main():
    """Main function to analyze a MATLAB .roi file."""
    # Get file path from user
    file_path = input("Enter the path to the MATLAB .roi file: ")
    
    # Analyze the file
    result = analyze_binary_file(file_path)
    
    if result and result['type'] == 'unknown':
        # Try to visualize as ROI coordinates
        print("\nTrying to visualize as ROI coordinates...")
        
        # Ask user if they want to try visualization
        try_viz = input("Would you like to try visualizing potential ROI coordinates? (y/n): ")
        if try_viz.lower() == 'y':
            # Try different offsets and formats
            offsets_to_try = [0, 4, 8, 16, 32, 64, 128]
            
            for offset in offsets_to_try:
                print(f"\nTrying offset {offset}...")
                
                # Try as int16
                print("Trying as int16 big-endian...")
                try_visualize_as_roi(result['data'], offset=offset, as_int16=True, big_endian=True)
                
                print("Trying as int16 little-endian...")
                try_visualize_as_roi(result['data'], offset=offset, as_int16=True, big_endian=False)
                
                # Try as float32
                print("Trying as float32 big-endian...")
                try_visualize_as_roi(result['data'], offset=offset, as_int16=False, big_endian=True)
                
                print("Trying as float32 little-endian...")
                try_visualize_as_roi(result['data'], offset=offset, as_int16=False, big_endian=False)
                
                # Ask if user wants to continue
                cont = input("Continue with next offset? (y/n): ")
                if cont.lower() != 'y':
                    break

if __name__ == '__main__':
    main()
