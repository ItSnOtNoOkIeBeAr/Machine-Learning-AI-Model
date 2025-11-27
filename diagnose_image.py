"""
Diagnose why an image has low confidence
"""

import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch.nn.functional as F
import os

HARDWARE_CLASSES = ['cpu', 'gpu', 'motherboard', 'psu', 'ram']

def diagnose_image(image_path):
    """Show detailed analysis of why confidence is low."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("\nLoading model...")
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=len(HARDWARE_CLASSES),
        ignore_mismatched_sizes=True
    )
    
    checkpoint = torch.load("models/best_vit_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    print("\n" + "="*70)
    print("📊 IMAGE DIAGNOSIS")
    print("="*70)
    print(f"Image: {image_path}")
    print(f"Size: {image.size[0]}x{image.size[1]} pixels")
    print(f"Format: {image.format}")
    
    # Process and predict
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)[0].cpu().numpy()
    
    # Show detailed predictions
    print("\n📈 DETAILED PREDICTIONS:")
    print("-"*70)
    sorted_indices = probabilities.argsort()[::-1]
    
    for i, idx in enumerate(sorted_indices, 1):
        class_name = HARDWARE_CLASSES[idx]
        confidence = probabilities[idx] * 100
        bar_length = int(probabilities[idx] * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        
        status = ""
        if i == 1:
            if confidence >= 60:
                status = " ✅ HIGH CONFIDENCE"
            elif confidence >= 40:
                status = " ⚠️ MEDIUM CONFIDENCE"
            else:
                status = " ❌ LOW CONFIDENCE"
        
        print(f"{i}. {class_name:12s} {bar} {confidence:6.2f}%{status}")
    
    # Analysis
    print("\n🔍 ANALYSIS:")
    print("-"*70)
    top_confidence = probabilities[sorted_indices[0]] * 100
    second_confidence = probabilities[sorted_indices[1]] * 100
    difference = top_confidence - second_confidence
    
    print(f"Top prediction confidence: {top_confidence:.2f}%")
    print(f"Confidence margin: {difference:.2f}% (difference from 2nd place)")
    
    if top_confidence < 40:
        print("\n❌ VERY LOW CONFIDENCE - Possible reasons:")
        print("   • Image quality is poor (blurry, dark, or pixelated)")
        print("   • Object is not clearly visible")
        print("   • Object is very different from training images")
        print("   • Image contains multiple objects or clutter")
        print("\n💡 SUGGESTIONS:")
        print("   • Take a clearer, well-lit photo of the hardware")
        print("   • Ensure the component fills most of the frame")
        print("   • Remove background clutter")
        print("   • Make sure lighting is even")
    elif top_confidence < 60:
        print("\n⚠️ MEDIUM CONFIDENCE - Possible reasons:")
        print("   • Image quality could be better")
        print("   • Object angle or lighting is unusual")
        print("   • Object looks similar to other hardware types")
        print("\n💡 SUGGESTIONS:")
        print("   • Try a different angle or lighting")
        print("   • Get closer to the component")
        print("   • Ensure background is clean")
    else:
        print("\n✅ HIGH CONFIDENCE - Image is clear and recognizable!")
        print("   Model is very confident in its prediction.")
    
    if difference < 10:
        print(f"\n⚠️ Small margin between top predictions ({difference:.2f}%)")
        print("   Model is confused between multiple classes!")
        print(f"   Top 2: {HARDWARE_CLASSES[sorted_indices[0]].upper()} vs {HARDWARE_CLASSES[sorted_indices[1]].upper()}")
        print("\n💡 This means:")
        print("   • The hardware might look similar to another type")
        print("   • Image doesn't show distinctive features clearly")
        print("   • Consider adding more training images of this type")
    
    # Compare to threshold
    print("\n🎯 CONFIDENCE THRESHOLD:")
    print("-"*70)
    print(f"Current threshold: 60%")
    print(f"Your image: {top_confidence:.2f}%")
    
    if top_confidence < 60:
        print(f"❌ Below threshold by {60 - top_confidence:.2f}%")
        print("\nOptions:")
        print("  1. Improve image quality (recommended)")
        print("  2. Lower threshold in model_setup.py (not recommended)")
        print("  3. Add more training images of this type")
    else:
        print(f"✅ Above threshold by {top_confidence - 60:.2f}%")
    
    print("="*70)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1].strip('"').strip("'")
        if not os.path.exists(image_path):
            print(f"❌ Error: Image not found at '{image_path}'")
        else:
            diagnose_image(image_path)
    else:
        print("Usage: python diagnose_image.py <path_to_image>")
        print("\nExample:")
        print('  python diagnose_image.py "C:\\Users\\jaype\\Downloads\\images.jpg"')
        print('  python diagnose_image.py dataset/val/ram/27_20451dd7b37542ed83d8a9f8ae15c47b.jpg')
