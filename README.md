# ROI Semi-Automatic Detection Pipeline

This directory contains scripts for the automatic ROI selection pipeline for dendrites and spines in calcium imaging data.

## Overview

The pipeline processes calcium imaging data stored as TIFF files and automatically detects regions of interest (ROIs) for dendrites and spines. It uses convolutional neural networks (CNNs) trained on manually annotated ROIs to perform the detection.

## Scripts

### 1. `run_roi_pipeline.py`

This script handles the initial data processing and preparation:

- Loads and preprocesses TIFF files
- Processes manual ROI annotations
- Prepares training data for the CNNs
- Visualizes ROI data

**Usage:**
```bash
python run_roi_pipeline.py --tiff-dir /path/to/tiff/files --roi-file /path/to/roi/file --output-dir ./output
```

**Arguments:**
- `--tiff-dir`: Directory containing TIFF files
- `--roi-file`: Path to ROI file
- `--output-dir`: Directory to save output
- `--start-idx`: Start index of files to process (default: 23)
- `--end-idx`: End index of files to process (default: 45)

### 2. `run_roi_cnn.py`

This script implements the CNN models for dendrite and spine detection:

- Trains the dendrite detection CNN
- Trains the spine detection CNN
- Applies the trained models to detect ROIs in new images

**Usage:**
```bash
# Train dendrite CNN
python run_roi_cnn.py train_dendrite --data-dir ./output/roi_pipeline/training_data/dendrite --output-dir ./output/dendrite_cnn

# Train spine CNN
python run_roi_cnn.py train_spine --data-dir ./output/roi_pipeline/training_data/spine --output-dir ./output/spine_cnn

# Detect ROIs in an image
python run_roi_cnn.py detect --image-path ./image.tif --dendrite-model ./output/dendrite_cnn/checkpoints/best_model.pth --spine-model ./output/spine_cnn/checkpoints/best_model.pth --output-dir ./output/detection
```

**Arguments for training:**
- `--data-dir`: Directory containing training data
- `--output-dir`: Directory to save output
- `--epochs`: Number of epochs to train for (default: 50)
- `--batch-size`: Batch size (default: 16)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--weight-decay`: Weight decay (default: 1e-5)
- `--val-split`: Validation split (default: 0.2)

**Arguments for detection:**
- `--image-path`: Path to image file
- `--dendrite-model`: Path to dendrite model
- `--spine-model`: Path to spine model
- `--output-dir`: Directory to save output
- `--dendrite-threshold`: Threshold for dendrite segmentation (default: 0.5)
- `--spine-threshold`: Threshold for spine segmentation (default: 0.5)

### 3. `run_full_roi_pipeline.py`

This script runs the complete pipeline, integrating all the steps:

1. Data preparation
2. Dendrite CNN training
3. Spine CNN training
4. ROI detection in test images

**Usage:**
```bash
python run_full_roi_pipeline.py --tiff-dir /path/to/tiff/files --roi-file /path/to/roi/file --output-dir ./output
```

**Arguments:**
- `--tiff-dir`: Directory containing TIFF files
- `--roi-file`: Path to ROI file
- `--output-dir`: Directory to save output
- `--start-idx`: Start index of files to process (default: 23)
- `--end-idx`: End index of files to process (default: 45)
- `--no-train`: Skip model training
- `--epochs`: Number of epochs for training (default: 50)
- `--batch-size`: Batch size for training (default: 16)

## Example Usage

To run the full pipeline on the provided data:

```bash
python run_full_roi_pipeline.py \
    --tiff-dir "/Volumes/TOSHIBA EXT/MT033/1/tiff" \
    --roi-file "/Volumes/TOSHIBA EXT/MT033/1/rois/MT033_01_2023-11-22_00001_00011_Turboreg.roi" \
    --output-dir "./output" \
    --start-idx 23 \
    --end-idx 45 \
    --epochs 50 \
    --batch-size 16
```

This will:
1. Process TIFF files from index 23 to 45
2. Use the provided ROI file for training
3. Train the dendrite and spine CNNs
4. Detect ROIs in the test images
5. Save all results to the `./output` directory

## Output Structure

The pipeline generates the following output structure:

```
output/
├── roi_pipeline/
│   ├── visualizations/
│   │   ├── roi_polygons.png
│   │   ├── roi_masks.png
│   │   ├── roi_classification.png
│   │   ├── spine_dendrite_associations.png
│   │   └── roi_properties.png
│   ├── training_data/
│   │   ├── dendrite/
│   │   │   ├── image_0000.npy
│   │   │   ├── mask_0000.npy
│   │   │   └── ...
│   │   └── spine/
│   │       ├── patch_0000.npy
│   │       ├── mask_0000.npy
│   │       └── ...
│   └── performance_report.txt
├── dendrite_cnn/
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   ├── final_model.pth
│   │   └── ...
│   ├── training_progress.png
│   ├── training_history.npy
│   └── training_report.txt
├── spine_cnn/
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   ├── final_model.pth
│   │   └── ...
│   ├── training_progress.png
│   ├── training_history.npy
│   └── training_report.txt
├── detection/
│   ├── image_000/
│   │   ├── image.png
│   │   ├── dendrite_mask.png
│   │   ├── spine_mask.png
│   │   ├── spine_dendrite_associations.png
│   │   ├── dendrite_mask.npy
│   │   └── spine_mask.npy
│   ├── image_001/
│   │   └── ...
│   └── ...
└── full_pipeline_report.txt
```

## Performance Metrics

The pipeline generates performance reports at each stage:

1. **ROI Pipeline Report**: Contains information about the preprocessing time, ROI processing time, and training data preparation time.
2. **CNN Training Reports**: Contain information about the training progress, including loss and Dice coefficient for both training and validation sets.
3. **Full Pipeline Report**: Contains information about the entire pipeline execution, including the time taken for each step.

## Requirements

- Python 3.12
- PyTorch
- NumPy
- SciPy
- scikit-image
- matplotlib
- tqdm

## Notes

- The pipeline is designed to process calcium imaging data of dendrites and spines.
- It uses a U-Net architecture with ResNet backbone for both dendrite and spine detection.
- The pipeline is memory-efficient, processing TIFF files in batches to manage memory usage.
- The detection accuracy can be adjusted by modifying the threshold parameters.
