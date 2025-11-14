"""
Unified AI System - Chat + Hardware Classification
- Chat Model: Pre-trained (ready to use)
- Vision Model: Train with your hardware images
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, ViTImageProcessor, ViTForImageClassification
import torch
from PIL import Image
import os

HARDWARE_CLASSES = ['cpu', 'gpu', 'motherboard', 'psu', 'ram']

def setup_chat_model(model_name="microsoft/phi-2"):
    """Load PRE-TRAINED chat model (no training needed)."""
    print(f"Loading chat model: {model_name}")
    print("This model is already trained and ready to use!")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"  # Automatically use GPU if available
    )
    
    print(f"✅ Chat model ready!")
    return model, tokenizer


def setup_vision_model(model_path="models/best_vit_model.pth"):
    """Load YOUR TRAINED vision model for hardware classification."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=len(HARDWARE_CLASSES),
        ignore_mismatched_sizes=True
    )
    
    # Load YOUR trained weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        val_acc = checkpoint.get('val_acc', 0)
        print(f"✅ Vision model loaded from {model_path}")
        print(f"   Validation accuracy: {val_acc:.2f}%")
    else:
        print(f"⚠️ WARNING: No trained model found at {model_path}")
        print(f"   Train first with: python train_vit_tiny.py")
        print(f"   Using untrained model (will give random results)")
    
    model.to(device)
    model.eval()
    
    return model, processor, device


def classify_hardware(image_path, model, processor, device):
    """Classify hardware from image."""
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_idx = logits.argmax(-1).item()
            confidence = probabilities[0][predicted_class_idx].item() * 100
        
        predicted_class = HARDWARE_CLASSES[predicted_class_idx]
        
        # Top 3 predictions
        top3_prob, top3_idx = torch.topk(probabilities[0], k=min(3, len(HARDWARE_CLASSES)))
        top3_predictions = []
        for prob, idx in zip(top3_prob, top3_idx):
            top3_predictions.append({
                'class': HARDWARE_CLASSES[idx.item()],
                'confidence': prob.item() * 100
            })
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top3': top3_predictions,
            'success': True
        }
    
    except Exception as e:
        return {'success': False, 'error': str(e)}


def chat_response(model, tokenizer, user_message, conversation_history="", max_length=200):
    """
    Generate a chat response from the model.
    
    Args:
        model: The loaded language model
        tokenizer: The tokenizer for the model
        user_message: The user's current message
        conversation_history: Previous conversation context
        max_length: Maximum length of generated response
    
    Returns:
        Generated response text
    """
    # Format the prompt with conversation history
    if conversation_history:
        prompt = f"{conversation_history}\nUser: {user_message}\nAssistant:"
    else:
        prompt = f"User: {user_message}\nAssistant:"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    # Move inputs to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    
    # Decode the response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "Assistant:" in full_response:
        response = full_response.split("Assistant:")[-1].strip()
    else:
        response = full_response[len(prompt):].strip()
    
    # Clean up response (remove any additional "User:" if generated)
    if "User:" in response:
        response = response.split("User:")[0].strip()
    
    return response


def unified_system():
    """Unified system with chat and hardware identification."""
    print("\n" + "="*80)
    print("🤖 UNIFIED AI SYSTEM - Chat + Hardware Identification")
    print("="*80)
    print("\n📚 How This Works:")
    print("  1️⃣ Chat Model (Phi-2): Already trained, ready to chat")
    print("  2️⃣ Vision Model: YOU trained this with hardware images")
    print("\nCommands:")
    print("  💬 Chat: Type your message")
    print("  🖼️ Identify: identify <image_path>")
    print("  ⚙️ Other: 'quit', 'clear', 'help', 'status'")
    print("="*80 + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"🖥️ Using device: {device} ({gpu_name})\n")
    
    print("Loading AI models...")
    chat_model, chat_tokenizer = setup_chat_model()
    vision_model, vision_processor, vision_device = setup_vision_model()
    
    conversation_history = ""
    vision_trained = os.path.exists("models/best_vit_model.pth")
    
    print("\n✅ System ready!\n")
    
    if not vision_trained:
        print("⚠️ NOTICE: Vision model not trained yet!")
        print("   Train it with: python train_vit_tiny.py\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            conversation_history = ""
            print("\n🧹 Conversation history cleared!\n")
            continue
        
        if user_input.lower() == 'status':
            print(f"\n📊 System Status:")
            print(f"   Chat Model: ✅ Ready (Pre-trained Phi-2)")
            print(f"   Vision Model: {'✅ Trained' if vision_trained else '⚠️ Not trained yet'}")
            print(f"   GPU: {gpu_name}")
            print()
            continue
        
        if user_input.lower() == 'help':
            print("\n💬 Chat Commands:")
            print("   - Type message to chat with AI")
            print("\n🖼️ Hardware Identification:")
            print("   - identify <path> - Classify hardware image")
            print("   - Example: identify dataset/val/cpu/image.jpg")
            print("\n⚙️ System Commands:")
            print("   - status - Check system status")
            print("   - clear - Reset conversation")
            print("   - quit - Exit system\n")
            continue
        
        if not user_input:
            continue
        
        # Hardware identification
        if user_input.lower().startswith('identify '):
            if not vision_trained:
                print("\n⚠️ Vision model not trained yet!")
                print("   Train first with: python train_vit_tiny.py\n")
                continue
            
            image_path = user_input[9:].strip().strip('"').strip("'")
            
            if not os.path.exists(image_path):
                print(f"\n❌ Image not found: {image_path}\n")
                continue
            
            print(f"\n🔍 Analyzing: {image_path}")
            print("⏳ Processing...\n")
            
            result = classify_hardware(image_path, vision_model, vision_processor, vision_device)
            
            if result['success']:
                predicted = result['predicted_class'].upper()
                confidence = result['confidence']
                
                print(f"🎯 Prediction: {predicted}")
                print(f"📊 Confidence: {confidence:.2f}%\n")
                print("📈 Top 3 Predictions:")
                for i, pred in enumerate(result['top3'], 1):
                    print(f"   {i}. {pred['class'].upper()}: {pred['confidence']:.2f}%")
                
                # AI explanation
                print(f"\n🤖 AI Explanation:")
                explanation_prompt = f"Briefly explain what a {predicted} is and its main function in a computer system in 2-3 sentences."
                explanation = chat_response(chat_model, chat_tokenizer, explanation_prompt, max_length=150)
                print(f"   {explanation}")
            else:
                print(f"❌ Error: {result['error']}")
            
            print()
            continue
        
        # Regular chat
        print("\n🤖 Assistant: ", end="", flush=True)
        
        try:
            response = chat_response(chat_model, chat_tokenizer, user_input, conversation_history)
            print(response)
            
            conversation_history += f"\nUser: {user_input}\nAssistant: {response}"
            
            history_parts = conversation_history.split("\nUser:")
            if len(history_parts) > 6:
                conversation_history = "\nUser:".join(history_parts[-6:])
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        print()


if __name__ == "__main__":
    unified_system()
