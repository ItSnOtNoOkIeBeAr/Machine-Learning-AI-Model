"""
Unified AI System - Auto Chat (Gemini + Phi-2 Fallback) + Hardware Classification
- Chat: Gemini (primary) with Phi-2 fallback
- Vision Model: Hardware component identification
- Confidence Threshold: 25% minimum to accept as hardware
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, ViTImageProcessor, ViTForImageClassification
import torch
import torch.nn.functional as F
from PIL import Image
import os

# Import Gemini
try:
    import google.generativeai as genai
    from config import GEMINI_API_KEY
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Gemini not available. Install: pip install google-generativeai")
except Exception as e:
    GEMINI_AVAILABLE = False
    print(f"⚠️ Gemini config error: {e}")

HARDWARE_CLASSES = ['cpu', 'gpu', 'motherboard', 'psu', 'ram']  # Match trained model (5 classes)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.25  # 25% minimum to accept as hardware

def setup_gemini():
    """Setup Google Gemini AI for better conversational responses."""
    if not GEMINI_AVAILABLE:
        return None
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("✅ Gemini AI ready! (Model: gemini-2.5-flash)")
        print("   Stable version supporting up to 1M tokens")
        return model
    except Exception as e:
        print(f"⚠️ Gemini setup failed: {e}")
        return None

def setup_chat_model(model_name="microsoft/phi-2"):
    """Load Phi-2 as fallback chat model."""
    print(f"Loading fallback chat model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    print(f"✅ Phi-2 fallback ready!")
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


def unified_chat_response(gemini_model, phi2_model, phi2_tokenizer, user_message, conversation_history=""):
    """
    Unified chat response using Gemini (primary) with Phi-2 fallback.
    Automatically tries Gemini first, falls back to Phi-2 if needed.
    
    Returns:
        tuple: (response_text, source) where source is 'predefined', 'gemini', or 'phi2'
    """
    # Pre-defined responses (instant, no API needed)
    predefined_responses = {
        "who made you": "I was created by a student at LSPU as part of a CSST 101 final project. I'm an AI system that combines chat capabilities with hardware identification!",
        "who created you": "I was created by a student at LSPU as part of a CSST 101 final project!",
        "what are you": "I'm an AI assistant powered by Gemini and Phi-2, with Vision Transformer for hardware identification!",
        "what can you do": "I can chat naturally about any topic and identify computer hardware from images. Type 'identify <image_path>' to classify components!",
        "hello": "Hello! 👋 I'm your AI assistant. Ask me anything or use 'identify <image_path>' to classify hardware!",
        "hi": "Hi there! 👋 I can chat about any topic and identify computer hardware. What can I help you with?",
        "help": "💬 Chat: Type your message\n🖼️ Identify: identify <path>\n⚙️ Commands: 'status', 'clear', 'quit'",
    }
    
    # Check for pre-defined responses
    user_lower = user_message.lower().strip()
    for question, answer in predefined_responses.items():
        if question in user_lower:
            return answer, "predefined"
    
    # Try Gemini first (if available)
    if gemini_model is not None:
        try:
            system_context = """You are a helpful AI assistant specializing in computer hardware and technology. 
Answer questions clearly and concisely in 2-4 sentences."""
            
            if conversation_history:
                full_prompt = f"{system_context}\n\n{conversation_history}\nUser: {user_message}"
            else:
                full_prompt = f"{system_context}\n\nUser: {user_message}"
            
            response = gemini_model.generate_content(full_prompt)
            return response.text.strip(), "gemini"
        
        except Exception as e:
            print(f"   [Gemini error, using Phi-2 fallback...]")
    
    # Fallback to Phi-2
    system_prompt = """You are a helpful assistant specializing in computer hardware. Answer questions clearly and concisely in 2-4 sentences."""
    
    if conversation_history:
        prompt = f"{system_prompt}\n\n{conversation_history}\nUser: {user_message}\nAssistant:"
    else:
        prompt = f"{system_prompt}\n\nUser: {user_message}\nAssistant:"
    
    inputs = phi2_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(phi2_model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = phi2_model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.3,
            pad_token_id=phi2_tokenizer.pad_token_id,
            eos_token_id=phi2_tokenizer.eos_token_id,
        )
    
    full_response = phi2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Assistant:" in full_response:
        response = full_response.split("Assistant:")[-1].strip()
    else:
        response = full_response.replace(prompt, "").strip()
    
    if "User:" in response:
        response = response.split("User:")[0].strip()
    
    sentences = response.split('.')
    if len(sentences) > 4:
        response = '.'.join(sentences[:4]) + '.'
    
    return response, "phi2"


def unified_system():
    """Unified system with automatic chat (Gemini + Phi-2) and hardware identification."""
    print("\n" + "="*80)
    print("🤖 UNIFIED AI SYSTEM - Auto Chat + Hardware Identification")
    print("="*80)
    print("\n📚 System Features:")
    print("  💬 Chat: Gemini (primary) with Phi-2 fallback")
    print("  🖼️ Vision: Hardware component identification")
    print(f"  🎯 Confidence Threshold: {CONFIDENCE_THRESHOLD*100:.0f}% minimum")
    print("\n💡 Commands:")
    print("  💬 Chat: Type your message")
    print("  🖼️ Identify: identify <image_path>")
    print("  ⚙️ Other: 'status', 'clear', 'help', 'quit'")
    print("="*80 + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"🖥️ Using device: {device} ({gpu_name})\n")
    
    print("Loading AI models...")
    
    # Load both models
    gemini_model = setup_gemini()
    phi2_model, phi2_tokenizer = setup_chat_model()
    vision_model, vision_processor, vision_device = setup_vision_model()
    
    gemini_available = gemini_model is not None
    conversation_history = ""
    vision_trained = os.path.exists("models/best_vit_model.pth")
    
    print("\n✅ System ready!\n")
    
    if gemini_available:
        print("💬 Chat: Using Gemini with Phi-2 fallback\n")
    else:
        print("💬 Chat: Using Phi-2 only (Gemini unavailable)\n")
    
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
            print(f"   Gemini API: {'✅ Connected (Primary)' if gemini_available else '❌ Not available'}")
            print(f"   Phi-2 Model: ✅ Loaded (Fallback)")
            print(f"   Vision Model: {'✅ Trained (63.49%)' if vision_trained else '⚠️ Not trained yet'}")
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
            response, source = unified_chat_response(
                gemini_model, phi2_model, phi2_tokenizer, 
                user_input, conversation_history
            )
            
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
