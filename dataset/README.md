# Dataset Structure

This folder contains the training and validation datasets for the hardware component classification model.

## Structure

```
dataset/
├── train/                    # Training images (70-80% of data)
│   ├── cpu/                 # CPU component images
│   ├── gpu/                 # GPU component images
│   ├── ram/                 # RAM module images
│   ├── motherboard/         # Motherboard images
│   └── psu/                 # Power Supply Unit images
│
└── val/                      # Validation images (20-30% of data)
    ├── cpu/
    ├── gpu/
    ├── ram/
    ├── motherboard/
    └── psu/
```

## Adding Your Images

### 1. Organize your images by component type
Place images in the appropriate category folders:
- **CPU**: Processors (Intel, AMD, etc.)
- **GPU**: Graphics cards
- **RAM**: Memory modules
- **Motherboard**: Mainboards
- **PSU**: Power supplies

### 2. Image Guidelines
- **Format**: JPG, PNG, or BMP
- **Size**: Any size (will be resized to 224x224)
- **Quality**: Clear, well-lit images work best
- **Quantity**: 
  - Minimum: 20-50 images per class
  - Recommended: 100+ images per class for better accuracy

### 3. Split Ratio
- **Training set** (`train/`): 70-80% of your images
- **Validation set** (`val/`): 20-30% of your images

Example:
- If you have 100 CPU images → 80 in `train/cpu/`, 20 in `val/cpu/`

## Example Dataset Structure

```
dataset/
├── train/
│   ├── cpu/
│   │   ├── cpu_001.jpg
│   │   ├── cpu_002.jpg
│   │   └── ... (80 images)
│   ├── gpu/
│   │   ├── gpu_001.jpg
│   │   └── ... (80 images)
│   └── ...
│
└── val/
    ├── cpu/
    │   ├── cpu_val_001.jpg
    │   └── ... (20 images)
    └── ...
```

## Data Collection Tips

### Sources for Images:
1. **Online retailers** (Amazon, Newegg, etc.)
2. **Manufacturer websites** (Intel, AMD, NVIDIA, etc.)
3. **Your own photos** of hardware components
4. **Image datasets** (Kaggle, Google Images, etc.)

### Tips:
- Include various angles and lighting conditions
- Mix different brands and models
- Ensure images show the component clearly
- Remove duplicate or very similar images

## After Adding Images

Once you've added your images:

1. **Check the dataset**:
   ```bash
   python -c "from torchvision import datasets; train_data = datasets.ImageFolder('dataset/train'); val_data = datasets.ImageFolder('dataset/val'); print(f'Train: {len(train_data)} images'); print(f'Val: {len(val_data)} images'); print(f'Classes: {train_data.classes}')"
   ```

2. **Start training**:
   ```bash
   python train_vit_tiny.py
   ```

## Current Status

📁 Folders created, waiting for images to be added!
