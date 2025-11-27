"""
Unified AI System - Chat + Hardware Classification
- Chat Model: Pre-trained (ready to use)
- Vision Model: Train with your hardware images
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, ViTImageProcessor, ViTForImageClassification
import torch
import torch.nn.functional as F
from PIL import Image
import os

HARDWARE_CLASSES = ['cpu', 'gpu', 'motherboard', 'psu', 'ram']  # Match trained model (5 classes)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.25  # 25% minimum to accept as hardware

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
    """Classify hardware from image with single confidence threshold."""
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            predicted_class_idx = logits.argmax(-1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        predicted_class = HARDWARE_CLASSES[predicted_class_idx]
        is_valid = confidence >= CONFIDENCE_THRESHOLD
        all_probs = probabilities[0].cpu().numpy()
        
        return predicted_class, confidence, is_valid, all_probs
    
    except Exception as e:
        return None, 0.0, False, None


def get_hardware_explanation(component):
    """Get a brief explanation of the hardware component."""
    explanations = {
        'cpu': "A CPU (Central Processing Unit) is the primary processor that executes\n   instructions and performs calculations. It acts as the brain of the\n   computer system.",
        'gpu': "A GPU (Graphics Processing Unit) is specialized for rendering graphics\n   and parallel processing. Essential for gaming, video editing, and AI tasks.",
        'ram': "RAM (Random Access Memory) provides temporary storage for active data\n   and programs. More RAM allows better multitasking and faster performance.",
        'motherboard': "A motherboard is the main circuit board connecting all components.\n   It houses the CPU, RAM, and provides connectivity for all other parts.",
        'psu': "A PSU (Power Supply Unit) converts AC power to DC power and distributes\n   it to all components. Critical for system stability and performance."
    }
    return explanations.get(component.lower(), "Hardware component detected.")


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
    # Pre-defined responses for common questions
    predefined_responses = {
        "who made you": "I was created by a student at LSPU as part of a CSST 101 final project. I'm an AI system that combines chat capabilities with hardware identification!",
        "who created you": "I was created by a student at LSPU as part of a CSST 101 final project. I'm an AI system that combines chat capabilities with hardware identification!",
        "who built you": "I was created by a student at LSPU as part of a CSST 101 final project. I'm an AI system that combines chat capabilities with hardware identification!",
        "what are you": "I'm an AI assistant built with Microsoft's Phi-2 language model and Vision Transformer. I can chat with you and identify computer hardware components from images!",
        "what can you do": "I can do two main things: 1) Chat with you about computer hardware and technology, and 2) Identify hardware components from images. Just type 'identify <image_path>' to classify a component!",
        "what is your purpose": "I'm designed to help identify computer hardware components and answer questions about PC hardware. I combine conversational AI with computer vision!",
        "how do you work": "I use two AI models: Microsoft Phi-2 for conversational responses and Vision Transformer for image classification. Together, they let me chat and identify hardware!",
        "what model are you": "I'm powered by Microsoft's Phi-2 language model (2.7B parameters) for chat and Google's Vision Transformer (ViT) for hardware image classification.",
        "hello": "Hello! 👋 I'm your hardware identification assistant. I can help identify computer components or answer questions about PC hardware. Try 'identify <image_path>' to classify a hardware image!",
        "hi": "Hi there! 👋 I'm your AI hardware assistant. I can chat about PC components and identify hardware from images. What would you like to know?",
        "hey": "Hey! 👋 I'm here to help with computer hardware questions and component identification. What can I do for you?",
        "help": "I can help you in two ways:\n1. Chat about computer hardware and technology\n2. Identify hardware components - type 'identify <path>' with an image path\n\nSupported components: CPU, GPU, RAM, Motherboard, PSU",
        "what hardware can you identify": "I can identify these computer components: CPU (processors), GPU (graphics cards), RAM (memory modules), Motherboards, and PSU (power supplies). Just use 'identify <image_path>'!",
        "how accurate are you": "My accuracy depends on image quality and training data. For best results, use clear, well-lit photos with the component as the main focus.",
        "thank you": "You're welcome! Feel free to ask more questions or identify more hardware components!",
        "thanks": "You're welcome! Let me know if you need anything else!"
    }
    
    # Check for pre-defined responses
    user_lower = user_message.lower().strip()
    for question, answer in predefined_responses.items():
        if question in user_lower:
            return answer
    
    # Add system prompt for better responses
    system_prompt = """You are a helpful assistant specializing in computer hardware. Answer questions clearly and concisely in 2-4 sentences. Focus on practical, accurate information."""
    
    # Format the prompt with conversation history
    if conversation_history:
        prompt = f"{system_prompt}\n\n{conversation_history}\nUser: {user_message}\nAssistant:"
    else:
        prompt = f"{system_prompt}\n\nUser: {user_message}\nAssistant:"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Move inputs to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response with improved parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,  # Reduced for more focused responses
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            top_k=50,  # Added to reduce randomness
            do_sample=True,
            repetition_penalty=1.3,  # Prevent repetitive text
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "Assistant:" in full_response:
        response = full_response.split("Assistant:")[-1].strip()
    else:
        response = full_response.replace(prompt, "").strip()
    
    # Clean up response (remove any additional "User:" if generated)
    if "User:" in response:
        response = response.split("User:")[0].strip()
    
    # Limit response length (keep max 4 sentences)
    sentences = response.split('.')
    if len(sentences) > 4:
        response = '.'.join(sentences[:4]) + '.'
    
    return response


def unified_system():
    """Unified system with chat and hardware identification."""
    print("\n" + "="*80)
    print("🤖 UNIFIED AI SYSTEM - Chat + Hardware Identification")
    print("="*80)
    print("\n📚 How This Works:")
    print("  1️⃣ Chat Model (Phi-2): Already trained, ready to chat")
    print("  2️⃣ Vision Model: YOU trained this with hardware images")
    print(f"  3️⃣ Confidence Threshold: {CONFIDENCE_THRESHOLD*100:.0f}% minimum for valid detection")
    print("\n💡 Commands:")
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
            print(f"   Vision Model: {'✅ Trained (77.36%)' if vision_trained else '⚠️ Not trained yet'}")
            print(f"   GPU: {gpu_name}")
            print(f"   Confidence Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
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
                print(f"\n❌ Error: Image not found at '{image_path}'\n")
                continue
            
            print(f"\n🔍 Analyzing: {image_path}")
            print("⏳ Processing...\n")
            
            predicted, confidence, is_valid, all_probs = classify_hardware(
                image_path, vision_model, vision_processor, vision_device
            )
            
            if predicted is None:
                print("❌ Error processing image\n")
                continue
            
            if is_valid:
                # Valid hardware detection (≥20% confidence)
                print(f"🎯 Prediction: {predicted.upper()}")
                print(f"📊 Confidence: {confidence*100:.2f}%\n")
                
                # Top 3 predictions
                top_3_indices = all_probs.argsort()[-3:][::-1]
                print("📈 Top 3 Predictions:")
                for i, idx in enumerate(top_3_indices, 1):
                    print(f"   {i}. {HARDWARE_CLASSES[idx].upper()}: {all_probs[idx]*100:.2f}%")
                
                # AI Explanation
                print(f"\n🤖 AI Explanation:")
                print(f"   {get_hardware_explanation(predicted)}")
                print()
            else:
                # Invalid - not hardware (< 20% confidence)
                print(f"🔴 NOT A HARDWARE COMPONENT")
                print(f"   Best guess: {predicted.upper()} ({confidence*100:.2f}%)")
                print(f"   ❌ Confidence too low (< {CONFIDENCE_THRESHOLD*100:.0f}%)\n")
                print("   This doesn't appear to be computer hardware!\n")
                
                # Show all predictions
                print("📊 All Predictions:")
                for i, class_name in enumerate(HARDWARE_CLASSES):
                    bar_length = int(all_probs[i] * 40)
                    bar = "█" * bar_length + "░" * (40 - bar_length)
                    print(f"   {class_name:12s} {bar} {all_probs[i]*100:5.2f}%")
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
