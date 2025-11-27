"""
Train Vision Transformer (ViT-Tiny) for Hardware Component Classification

This script trains a Vision Transformer model to classify computer hardware components:
- CPU
- GPU
- RAM
- Motherboard
- PSU (Power Supply Unit)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor
import os
from tqdm import tqdm

# Configuration
CONFIG = {
    'model_name': 'google/vit-base-patch16-224-in21k',  # Pre-trained ViT model
    'num_classes': 5,  # cpu, gpu, ram, motherboard, psu
    'batch_size': 32,  # Increased for GPU (adjust if OOM)
    'num_epochs': 20,  # Increased from 10 for better convergence
    'learning_rate': 2e-5,
    'train_dir': 'dataset/train',
    'val_dir': 'dataset/val',
    'save_dir': 'models',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'use_mixed_precision': True,  # Use FP16 for faster training on Turing GPUs
    'num_workers': 4  # Parallel data loading
}

# Class names
CLASS_NAMES = ['cpu', 'gpu', 'ram', 'motherboard', 'psu']


def get_transforms():
    """Define data transformations for training and validation."""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),  # Less aggressive crop
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  # Moderate rotation
        transforms.ColorJitter(
            brightness=0.2,  # Balanced color variations
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_dataloaders():
    """Create train and validation dataloaders."""
    train_transform, val_transform = get_transforms()
    
    train_dataset = datasets.ImageFolder(CONFIG['train_dir'], transform=train_transform)
    val_dataset = datasets.ImageFolder(CONFIG['val_dir'], transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if CONFIG['device'] == 'cuda' else False  # Faster data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    
    return train_loader, val_loader, train_dataset.classes


def create_model():
    """Create and configure the ViT model."""
    model = ViTForImageClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=CONFIG['num_classes'],
        ignore_mismatched_sizes=True
    )
    return model.to(CONFIG['device'])


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images).logits
                loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Use mixed precision for inference too
            with torch.cuda.amp.autocast():
                outputs = model(images).logits
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def main():
    """Main training function."""
    print("=" * 60)
    print("Hardware Component Classification - ViT Training")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Mixed Precision: {'Enabled (FP16)' if CONFIG['use_mixed_precision'] else 'Disabled'}")
    print(f"Model: {CONFIG['model_name']}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print(f"Learning rate: {CONFIG['learning_rate']}")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(CONFIG['train_dir']) or not os.path.exists(CONFIG['val_dir']):
        print("\n❌ Error: Dataset directories not found!")
        print(f"Please add images to:")
        print(f"  - {CONFIG['train_dir']}")
        print(f"  - {CONFIG['val_dir']}")
        return
    
    # Create save directory
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, classes = create_dataloaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Classes: {classes}")
    
    # Create model
    print("\nInitializing model...")
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Initialize mixed precision scaler for GPU
    scaler = None
    if CONFIG['device'] == 'cuda' and CONFIG['use_mixed_precision']:
        scaler = torch.cuda.amp.GradScaler()
        print("✅ Mixed precision (FP16) training enabled for faster GPU training!")
    
    # Training loop
    best_val_acc = 0.0
    
    print("\nStarting training...")
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, CONFIG['device'], scaler)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(CONFIG['save_dir'], 'best_vit_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': classes
            }, save_path)
            print(f"  ✅ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "=" * 60)
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {CONFIG['save_dir']}/best_vit_model.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()
