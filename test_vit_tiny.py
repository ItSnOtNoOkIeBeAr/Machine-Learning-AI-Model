"""
Test Vision Transformer (ViT) for Hardware Component Classification

This script tests the trained ViT model on individual images or test datasets.
"""

import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
import os
import sys

# Configuration
CONFIG = {
    'model_path': 'models/best_vit_model.pth',
    'num_classes': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Class names
CLASS_NAMES = ['cpu', 'gpu', 'motherboard', 'psu', 'ram']


def get_transform():
    """Get image transformation for inference."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_model():
    """Load the trained model."""
    print("Loading model...")
    
    if not os.path.exists(CONFIG['model_path']):
        print(f"❌ Error: Model not found at {CONFIG['model_path']}")
        print("Please train the model first using train_vit_tiny.py")
        sys.exit(1)
    
    # Create model
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=CONFIG['num_classes'],
        ignore_mismatched_sizes=True
    )
    
    # Load checkpoint
    checkpoint = torch.load(CONFIG['model_path'], map_location=CONFIG['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(CONFIG['device'])
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Best validation accuracy: {checkpoint['val_acc']:.2f}%")
    print(f"   Classes: {checkpoint.get('classes', CLASS_NAMES)}")
    
    return model, checkpoint.get('classes', CLASS_NAMES)


def predict_image(model, image_path, classes, transform):
    """Predict the class of a single image."""
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None
    
    # Transform and add batch dimension
    image_tensor = transform(image).unsqueeze(0).to(CONFIG['device'])
    
    # Predict with mixed precision for faster inference on GPU
    with torch.no_grad():
        if CONFIG['device'] == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(image_tensor).logits
        else:
            outputs = model(image_tensor).logits
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = classes[predicted.item()]
    confidence_score = confidence.item() * 100
    
    # Get top 3 predictions
    top_probs, top_indices = torch.topk(probabilities, min(3, len(classes)))
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence_score,
        'top_predictions': [
            (classes[idx], prob.item() * 100)
            for idx, prob in zip(top_indices[0].tolist(), top_probs[0].tolist())
        ]
    }


def test_single_image(image_path):
    """Test a single image."""
    print("=" * 60)
    print("Hardware Component Classification - Testing")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Image: {image_path}")
    print("=" * 60)
    
    # Load model
    model, classes = load_model()
    transform = get_transform()
    
    # Predict
    print("\nMaking prediction...")
    result = predict_image(model, image_path, classes, transform)
    
    if result:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Predicted Class: {result['predicted_class'].upper()}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print("\nTop 3 Predictions:")
        for i, (cls, prob) in enumerate(result['top_predictions'], 1):
            print(f"  {i}. {cls.upper()}: {prob:.2f}%")
        print("=" * 60)


def test_directory(directory_path):
    """Test all images in a directory."""
    print("=" * 60)
    print("Hardware Component Classification - Batch Testing")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    print(f"Directory: {directory_path}")
    print("=" * 60)
    
    # Load model
    model, classes = load_model()
    transform = get_transform()
    
    # Get all image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [
        f for f in os.listdir(directory_path)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    
    if not image_files:
        print(f"❌ No image files found in {directory_path}")
        return
    
    print(f"\nFound {len(image_files)} images")
    print("\nTesting images...\n")
    
    # Test each image
    results = []
    for i, filename in enumerate(image_files, 1):
        image_path = os.path.join(directory_path, filename)
        result = predict_image(model, image_path, classes, transform)
        
        if result:
            results.append((filename, result))
            print(f"{i}. {filename}")
            print(f"   → {result['predicted_class'].upper()} ({result['confidence']:.2f}%)")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Tested {len(results)} images")
    print("=" * 60)


def interactive_mode():
    """Interactive testing mode."""
    print("=" * 60)
    print("Hardware Component Classification - Interactive Mode")
    print("=" * 60)
    
    # Load model
    model, classes = load_model()
    transform = get_transform()
    
    print("\nEnter image path to classify (or 'quit' to exit)")
    
    while True:
        image_path = input("\nImage path: ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            continue
        
        result = predict_image(model, image_path, classes, transform)
        
        if result:
            print(f"\n→ Predicted: {result['predicted_class'].upper()} ({result['confidence']:.2f}%)")
            print("  Top 3:")
            for cls, prob in result['top_predictions']:
                print(f"    - {cls.upper()}: {prob:.2f}%")


def main():
    """Main testing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ViT model on hardware components')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--directory', type=str, help='Path to directory of images')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.image:
        test_single_image(args.image)
    elif args.directory:
        test_directory(args.directory)
    elif args.interactive:
        interactive_mode()
    else:
        print("Usage:")
        print("  Single image:  python test_vit_tiny.py --image path/to/image.jpg")
        print("  Directory:     python test_vit_tiny.py --directory path/to/images")
        print("  Interactive:   python test_vit_tiny.py --interactive")


if __name__ == "__main__":
    main()
