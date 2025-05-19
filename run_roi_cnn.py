#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI CNN Models for Dendrite and Spine Detection

This script implements the CNN models for dendrite and spine detection.
It trains the models on the prepared training data and applies them to detect ROIs.

Created on Sat May 18 15:05:00 2025

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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.models.unet import UNetWithResNetBackbone
from src.models.losses import dice_loss, bce_dice_loss
from src.utils.tiff_handling import (
    load_tiff_stack, create_max_projection, preprocess_frame,
    split_image_into_patches, filter_patches_with_dendrites
)
from src.utils.visualization import (
    visualize_model_predictions, create_training_progress_plot
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


class ROIDataset(Dataset):
    """Dataset for ROI detection."""
    
    def __init__(self, data_dir: str, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing images and masks
            transform: Optional transform to apply to the data
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Find all image files
        self.image_files = sorted(glob(os.path.join(data_dir, 'image_*.npy')))
        self.mask_files = sorted(glob(os.path.join(data_dir, 'mask_*.npy')))
        
        # Check if patch dataset
        if not self.image_files:
            self.image_files = sorted(glob(os.path.join(data_dir, 'patch_*.npy')))
            self.mask_files = sorted(glob(os.path.join(data_dir, 'mask_*.npy')))
        
        # Verify that we have the same number of images and masks
        if len(self.image_files) != len(self.mask_files):
            raise ValueError(f"Number of images ({len(self.image_files)}) does not match number of masks ({len(self.mask_files)})")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, mask)
        """
        # Load image and mask
        image = np.load(self.image_files[idx])
        mask = np.load(self.mask_files[idx])
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension
        mask = torch.from_numpy(mask).float().unsqueeze(0)  # Add channel dimension
        
        # Apply transform if provided
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask


@measure_time
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 50,
    save_dir: str = './checkpoints'
) -> Dict[str, List[float]]:
    """
    Train the model.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use for training
        num_epochs: Number of epochs to train for
        save_dir: Directory to save checkpoints
        
    Returns:
        Dictionary containing training history
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': []
    }
    
    # Initialize best validation loss
    best_val_loss = float('inf')
    
    # Train the model
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * images.size(0)
            train_dice += (1 - dice_loss(outputs, masks).item()) * images.size(0)
        
        # Calculate average loss and dice
        train_loss = train_loss / len(train_loader.dataset)
        train_dice = train_dice / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
                # Move data to device
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # Update statistics
                val_loss += loss.item() * images.size(0)
                val_dice += (1 - dice_loss(outputs, masks).item()) * images.size(0)
        
        # Calculate average loss and dice
        val_loss = val_loss / len(val_loader.dataset)
        val_dice = val_dice / len(val_loader.dataset)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"Saved checkpoint (epoch {epoch+1}, val_loss: {val_loss:.4f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    
    return history


@measure_time
def predict(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    patch_size: int = 128,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Apply the model to predict ROIs in an image.
    
    Args:
        model: Trained model
        image: Image to predict on
        device: Device to use for prediction
        patch_size: Size of patches to use
        threshold: Threshold for binary segmentation
        
    Returns:
        Binary mask of predicted ROIs
    """
    # Set model to evaluation mode
    model.eval()
    
    # Check if image is large enough for patching
    if min(image.shape) < patch_size:
        # Resize image to minimum size
        from skimage.transform import resize
        image = resize(image, (max(patch_size, image.shape[0]), max(patch_size, image.shape[1])))
    
    # Split image into patches
    patches, coordinates = split_image_into_patches(image, patch_size=patch_size)
    
    # Create output mask
    output_mask = np.zeros(image.shape, dtype=np.float32)
    count_mask = np.zeros(image.shape, dtype=np.float32)
    
    # Process each patch
    with torch.no_grad():
        for patch, (y, x) in zip(patches, coordinates):
            # Convert patch to tensor
            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            
            # Move to device
            patch_tensor = patch_tensor.to(device)
            
            # Predict
            output = model(patch_tensor)
            
            # Convert output to numpy
            output = output.cpu().numpy().squeeze()
            
            # Add to output mask
            output_mask[y:y+patch_size, x:x+patch_size] += output
            count_mask[y:y+patch_size, x:x+patch_size] += 1
    
    # Average overlapping regions
    output_mask = np.divide(output_mask, count_mask, where=count_mask > 0)
    
    # Apply threshold
    binary_mask = (output_mask > threshold).astype(np.uint8)
    
    return binary_mask


def train_dendrite_cnn(
    data_dir: str,
    output_dir: str,
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    val_split: float = 0.2
) -> None:
    """
    Train the dendrite detection CNN.
    
    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save output
        num_epochs: Number of epochs to train for
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        val_split: Validation split
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = ROIDataset(data_dir)
    
    # Split dataset into train and validation sets
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=val_split, random_state=42
    )
    
    # Create data loaders
    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_indices)
    )
    val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_indices)
    )
    
    # Create model
    model = UNetWithResNetBackbone(in_channels=1, out_channels=1)
    model = model.to(device)
    
    # Create loss function and optimizer
    criterion = bce_dice_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Train model
    print("Training dendrite detection CNN...")
    history, training_time = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=num_epochs, save_dir=os.path.join(output_dir, 'checkpoints')
    )
    
    # Create training progress plot
    create_training_progress_plot(
        history,
        save_path=os.path.join(output_dir, 'training_progress.png')
    )
    
    # Save training history
    np.save(os.path.join(output_dir, 'training_history.npy'), history)
    
    # Generate training report
    report_path = os.path.join(output_dir, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write("Dendrite CNN Training Report\n")
        f.write("===========================\n\n")
        f.write(f"Training Data: {data_dir}\n")
        f.write(f"Number of Samples: {len(dataset)}\n")
        f.write(f"Train/Val Split: {1-val_split:.2f}/{val_split:.2f}\n\n")
        
        f.write("Training Parameters:\n")
        f.write(f"  Epochs: {num_epochs}\n")
        f.write(f"  Batch Size: {batch_size}\n")
        f.write(f"  Learning Rate: {learning_rate}\n")
        f.write(f"  Weight Decay: {weight_decay}\n\n")
        
        f.write("Training Results:\n")
        f.write(f"  Final Train Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss: {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Train Dice: {history['train_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Dice: {history['val_dice'][-1]:.4f}\n\n")
        
        f.write(f"Training Time: {training_time:.2f} seconds\n")
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Results saved to: {output_dir}")
    print(f"Training report: {report_path}")


def train_spine_cnn(
    data_dir: str,
    output_dir: str,
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    val_split: float = 0.2
) -> None:
    """
    Train the spine detection CNN.
    
    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save output
        num_epochs: Number of epochs to train for
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        val_split: Validation split
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = ROIDataset(data_dir)
    
    # Split dataset into train and validation sets
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=val_split, random_state=42
    )
    
    # Create data loaders
    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_indices)
    )
    val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_indices)
    )
    
    # Create model (smaller U-Net for spine detection)
    model = UNetWithResNetBackbone(in_channels=1, out_channels=1, backbone='resnet18')
    model = model.to(device)
    
    # Create loss function and optimizer
    criterion = bce_dice_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Train model
    print("Training spine detection CNN...")
    history, training_time = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=num_epochs, save_dir=os.path.join(output_dir, 'checkpoints')
    )
    
    # Create training progress plot
    create_training_progress_plot(
        history,
        save_path=os.path.join(output_dir, 'training_progress.png')
    )
    
    # Save training history
    np.save(os.path.join(output_dir, 'training_history.npy'), history)
    
    # Generate training report
    report_path = os.path.join(output_dir, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write("Spine CNN Training Report\n")
        f.write("========================\n\n")
        f.write(f"Training Data: {data_dir}\n")
        f.write(f"Number of Samples: {len(dataset)}\n")
        f.write(f"Train/Val Split: {1-val_split:.2f}/{val_split:.2f}\n\n")
        
        f.write("Training Parameters:\n")
        f.write(f"  Epochs: {num_epochs}\n")
        f.write(f"  Batch Size: {batch_size}\n")
        f.write(f"  Learning Rate: {learning_rate}\n")
        f.write(f"  Weight Decay: {weight_decay}\n\n")
        
        f.write("Training Results:\n")
        f.write(f"  Final Train Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss: {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Train Dice: {history['train_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Dice: {history['val_dice'][-1]:.4f}\n\n")
        
        f.write(f"Training Time: {training_time:.2f} seconds\n")
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Results saved to: {output_dir}")
    print(f"Training report: {report_path}")


@measure_time
def detect_rois(
    image: np.ndarray,
    dendrite_model_path: str,
    spine_model_path: str,
    output_dir: str,
    dendrite_threshold: float = 0.5,
    spine_threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Detect ROIs in an image.
    
    Args:
        image: Image to detect ROIs in
        dendrite_model_path: Path to dendrite model
        spine_model_path: Path to spine model
        output_dir: Directory to save output
        dendrite_threshold: Threshold for dendrite segmentation
        spine_threshold: Threshold for spine segmentation
        
    Returns:
        Dictionary containing detected ROIs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dendrite model
    dendrite_model = UNetWithResNetBackbone(in_channels=1, out_channels=1)
    dendrite_model.load_state_dict(torch.load(dendrite_model_path, map_location=device))
    dendrite_model = dendrite_model.to(device)
    
    # Load spine model
    spine_model = UNetWithResNetBackbone(in_channels=1, out_channels=1, backbone='resnet18')
    spine_model.load_state_dict(torch.load(spine_model_path, map_location=device))
    spine_model = spine_model.to(device)
    
    # Preprocess image
    preprocessed = preprocess_frame(image)
    
    # Detect dendrites
    print("Detecting dendrites...")
    dendrite_mask, dendrite_time = predict(
        dendrite_model, preprocessed, device, threshold=dendrite_threshold
    )
    
    # Extract patches containing dendrites
    patches, coordinates = split_image_into_patches(preprocessed)
    filtered_patches, filtered_coordinates = filter_patches_with_dendrites(
        patches, coordinates, dendrite_mask, threshold=0.05
    )
    
    # Detect spines
    print("Detecting spines...")
    spine_mask = np.zeros_like(dendrite_mask)
    
    for patch, (y, x) in zip(filtered_patches, filtered_coordinates):
        # Predict spines in patch
        patch_spine_mask = predict(
            spine_model, patch, device, threshold=spine_threshold
        )[0]  # Extract mask from tuple (mask, time)
        
        # Add to spine mask
        spine_mask[y:y+128, x:x+128] = np.logical_or(
            spine_mask[y:y+128, x:x+128], patch_spine_mask
        )
    
    # Associate spines with dendrites
    from src.utils.roi_processing import associate_spines_with_dendrites
    from skimage.measure import label, regionprops
    
    # Label connected components
    labeled_dendrites = label(dendrite_mask)
    labeled_spines = label(spine_mask)
    
    # Get individual masks
    dendrite_masks = []
    for i in range(1, labeled_dendrites.max() + 1):
        dendrite_masks.append((labeled_dendrites == i).astype(np.uint8))
    
    spine_masks = []
    for i in range(1, labeled_spines.max() + 1):
        spine_masks.append((labeled_spines == i).astype(np.uint8))
    
    # Associate spines with dendrites
    associations = associate_spines_with_dendrites(spine_masks, dendrite_masks)
    
    # Visualize results
    from src.utils.visualization import (
        visualize_roi_masks, visualize_spine_dendrite_associations
    )
    
    # Visualize dendrite mask
    visualize_roi_masks(
        image, dendrite_masks,
        save_path=os.path.join(output_dir, 'dendrite_mask.png')
    )
    
    # Visualize spine mask
    visualize_roi_masks(
        image, spine_masks,
        save_path=os.path.join(output_dir, 'spine_mask.png')
    )
    
    # Visualize spine-dendrite associations
    visualize_spine_dendrite_associations(
        image, dendrite_masks, spine_masks, associations,
        save_path=os.path.join(output_dir, 'spine_dendrite_associations.png')
    )
    
    # Save results
    np.save(os.path.join(output_dir, 'dendrite_mask.npy'), dendrite_mask)
    np.save(os.path.join(output_dir, 'spine_mask.npy'), spine_mask)
    
    # Return results
    return {
        'dendrite_mask': dendrite_mask,
        'spine_mask': spine_mask,
        'dendrite_masks': dendrite_masks,
        'spine_masks': spine_masks,
        'associations': associations
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='ROI CNN Models for Dendrite and Spine Detection')
    subparsers = parser.add_subparsers(dest='mode', help='Mode to run')
    
    # Create parser for dendrite training mode
    dendrite_parser = subparsers.add_parser('train_dendrite', help='Train dendrite detection CNN')
    dendrite_parser.add_argument('--data-dir', type=str, required=True,
                                help='Directory containing training data')
    dendrite_parser.add_argument('--output-dir', type=str, required=True,
                                help='Directory to save output')
    dendrite_parser.add_argument('--epochs', type=int, default=50,
                                help='Number of epochs to train for')
    dendrite_parser.add_argument('--batch-size', type=int, default=16,
                                help='Batch size')
    dendrite_parser.add_argument('--learning-rate', type=float, default=1e-4,
                                help='Learning rate')
    dendrite_parser.add_argument('--weight-decay', type=float, default=1e-5,
                                help='Weight decay')
    dendrite_parser.add_argument('--val-split', type=float, default=0.2,
                                help='Validation split')
    
    # Create parser for spine training mode
    spine_parser = subparsers.add_parser('train_spine', help='Train spine detection CNN')
    spine_parser.add_argument('--data-dir', type=str, required=True,
                            help='Directory containing training data')
    spine_parser.add_argument('--output-dir', type=str, required=True,
                            help='Directory to save output')
    spine_parser.add_argument('--epochs', type=int, default=50,
                            help='Number of epochs to train for')
    spine_parser.add_argument('--batch-size', type=int, default=16,
                            help='Batch size')
    spine_parser.add_argument('--learning-rate', type=float, default=1e-4,
                            help='Learning rate')
    spine_parser.add_argument('--weight-decay', type=float, default=1e-5,
                            help='Weight decay')
    spine_parser.add_argument('--val-split', type=float, default=0.2,
                            help='Validation split')
    
    # Create parser for detection mode
    detect_parser = subparsers.add_parser('detect', help='Detect ROIs in an image')
    detect_parser.add_argument('--image-path', type=str, required=True,
                            help='Path to image file')
    detect_parser.add_argument('--dendrite-model', type=str, required=True,
                            help='Path to dendrite model')
    detect_parser.add_argument('--spine-model', type=str, required=True,
                            help='Path to spine model')
    detect_parser.add_argument('--output-dir', type=str, required=True,
                            help='Directory to save output')
    detect_parser.add_argument('--dendrite-threshold', type=float, default=0.5,
                            help='Threshold for dendrite segmentation')
    detect_parser.add_argument('--spine-threshold', type=float, default=0.5,
                            help='Threshold for spine segmentation')
    
    args = parser.parse_args()
    
    # Run the selected mode
    if args.mode == 'train_dendrite':
        train_dendrite_cnn(
            args.data_dir,
            args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            val_split=args.val_split
        )
    elif args.mode == 'train_spine':
        train_spine_cnn(
            args.data_dir,
            args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            val_split=args.val_split
        )
    elif args.mode == 'detect':
        # Load image
        from skimage import io
        image = io.imread(args.image_path)
        
        # Detect ROIs
        detect_rois(
            image,
            args.dendrite_model,
            args.spine_model,
            args.output_dir,
            dendrite_threshold=args.dendrite_threshold,
            spine_threshold=args.spine_threshold
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
