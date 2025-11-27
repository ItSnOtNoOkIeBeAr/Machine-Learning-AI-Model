"""
Auto-split dataset into train and validation sets
This script will copy images from train/ to val/ folders automatically
"""

import os
import shutil
from pathlib import Path
import random

# Configuration
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
VAL_SPLIT = 0.2  # 20% for validation, 80% for training

CLASSES = ['cpu', 'gpu', 'ram', 'motherboard', 'psu']

def ensure_directories():
    """Create all necessary directories if they don't exist."""
    for class_name in CLASSES:
        train_class_dir = Path(TRAIN_DIR) / class_name
        val_class_dir = Path(VAL_DIR) / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)

def split_dataset():
    """Split training images into train and validation sets."""
    print("\n" + "=" * 70)
    print("Dataset Splitter - Auto Split Train/Val")
    print("=" * 70)
    print(f"Validation split: {VAL_SPLIT*100:.0f}%")
    print("=" * 70)
    
    # Ensure directories exist
    ensure_directories()
    
    total_moved = 0
    
    for class_name in CLASSES:
        train_path = Path(TRAIN_DIR) / class_name
        val_path = Path(VAL_DIR) / class_name
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        all_images = [
            f for f in train_path.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        total_images = len(all_images)
        
        if total_images == 0:
            print(f"\n⚠️  {class_name}: No images found - SKIPPING")
            continue
        
        # Calculate how many to move to validation
        num_val = max(1, int(total_images * VAL_SPLIT))  # At least 1 for validation
        
        # If only 1 image total, can't split
        if total_images == 1:
            print(f"\n⚠️  {class_name}: Only 1 image - need at least 2 to split!")
            print(f"    Please add more images to train/{class_name}/")
            continue
        
        # Randomly select images for validation
        random.shuffle(all_images)
        val_images = all_images[:num_val]
        
        # Move images to validation folder
        moved = 0
        for img in val_images:
            dest = val_path / img.name
            try:
                shutil.move(str(img), str(dest))
                moved += 1
            except Exception as e:
                print(f"    Error moving {img.name}: {e}")
        
        train_remaining = total_images - moved
        
        print(f"\n✅ {class_name}:")
        print(f"    Total: {total_images} images")
        print(f"    Train: {train_remaining} images ({100*(1-VAL_SPLIT):.0f}%)")
        print(f"    Val:   {moved} images ({VAL_SPLIT*100:.0f}%)")
        
        total_moved += moved
    
    print("\n" + "=" * 70)
    if total_moved > 0:
        print(f"✅ Successfully split dataset!")
        print(f"   Moved {total_moved} images to validation folders")
    else:
        print("⚠️  No images were moved. Please add more images to train folders.")
    print("=" * 70)


def check_dataset():
    """Display current dataset statistics."""
    print("\n" + "=" * 70)
    print("Current Dataset Statistics:")
    print("=" * 70)
    
    # Ensure directories exist first
    ensure_directories()
    
    train_total = 0
    val_total = 0
    
    for class_name in CLASSES:
        train_path = Path(TRAIN_DIR) / class_name
        val_path = Path(VAL_DIR) / class_name
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        
        train_images = len([
            f for f in train_path.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]) if train_path.exists() else 0
        
        val_images = len([
            f for f in val_path.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]) if val_path.exists() else 0
        
        total = train_images + val_images
        train_total += train_images
        val_total += val_images
        
        print(f"\n{class_name}:")
        print(f"  Train: {train_images} images")
        print(f"  Val:   {val_images} images")
        print(f"  Total: {total} images")
    
    print("\n" + "=" * 70)
    print(f"Grand Total:")
    print(f"  Train: {train_total} images")
    print(f"  Val:   {val_total} images")
    print(f"  Total: {train_total + val_total} images")
    print("=" * 70)
    
    # Check if ready to train
    if val_total == 0:
        print("\n⚠️  No validation images! Cannot train yet.")
        print("   Run: python split_dataset.py --split")
    elif train_total < 5 * len(CLASSES):
        print("\n⚠️  Warning: Very few training images!")
        print("   Recommended: At least 20-50 images per class for good results.")
    else:
        print("\n✅ Dataset is ready for training!")


if __name__ == "__main__":
    import sys
    
    print("\nOptions:")
    print("  1. Check dataset only")
    print("  2. Auto-split dataset (move images from train/ to val/)")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--split":
        split_dataset()
    elif len(sys.argv) > 1 and sys.argv[1] == "--check":
        check_dataset()
    else:
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            check_dataset()
        elif choice == "2":
            confirm = input("\nThis will move 20% of images from train/ to val/. Continue? (y/n): ")
            if confirm.lower() == 'y':
                split_dataset()
                check_dataset()
            else:
                print("Cancelled.")
        else:
            print("Invalid choice.")
