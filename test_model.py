"""
Quick test script to verify the model works correctly
"""

import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch.nn.functional as F
import os

HARDWARE_CLASSES = ['cpu', 'gpu', 'motherboard', 'psu', 'ram']
CONFIDENCE_THRESHOLD = 0.60

def test_model():
    """Test the trained model with sample images."""
    print("="*70)
    print("Testing Hardware Classification Model")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=len(HARDWARE_CLASSES),
        ignore_mismatched_sizes=True
    )
    
    model_path = "models/best_vit_model.pth"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded! Validation accuracy: {checkpoint['val_acc']:.2f}%")
    else:
        print("❌ Model not found!")
        return
    
    model.to(device)
    model.eval()
    
    # Test with sample images from each class
    print("\n" + "="*70)
    print("Testing with sample images:")
    print("="*70)
    
    # Test with validation images (better confidence)
    test_images = [
        ("dataset/val/cpu/4_dc406a3527654fe2bb6ad60ba284f606.jpg", "cpu"),
        ("dataset/val/gpu/17_876d9659712c4d17990012c207fe7b55.jpg", "gpu"),
        ("dataset/val/ram/27_20451dd7b37542ed83d8a9f8ae15c47b.jpg", "ram"),
        ("dataset/val/motherboard/7_73516d1ecf0046cbac16decae290e6a1.jpg", "motherboard"),
    ]
    
    for image_path, expected_class in test_images:
        if not os.path.exists(image_path):
            print(f"\n⚠️ Skipping {expected_class} - image not found")
            continue
        
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
                predicted_idx = logits.argmax(-1).item()
                confidence = probabilities[0][predicted_idx].item()
            
            predicted_class = HARDWARE_CLASSES[predicted_idx]
            is_correct = predicted_class == expected_class
            is_valid = confidence >= CONFIDENCE_THRESHOLD
            
            status = "✅" if is_correct else "❌"
            validity = "✅ VALID" if is_valid else "⚠️ LOW CONF"
            
            print(f"\n{status} Expected: {expected_class.upper():12s} | Predicted: {predicted_class.upper():12s} | {confidence*100:5.2f}% {validity}")
            
        except Exception as e:
            print(f"\n❌ Error testing {expected_class}: {e}")
    
    print("\n" + "="*70)
    print("Test completed!")
    print("="*70)

if __name__ == "__main__":
    test_model()
