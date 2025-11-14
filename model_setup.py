"""
AI Model Setup - PyTorch with Hugging Face Transformers
This script demonstrates loading and using a modern transformer model (~3GB)
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def setup_model(model_name="microsoft/phi-2"):
    """
    Load a modern transformer model from Hugging Face.
    
    Default model: Microsoft Phi-2 (~3GB)
    - Efficient 2.7B parameter model
    - Good performance for its size
    - Suitable for text generation and understanding tasks
    
    Other options you can try:
    - "google/flan-t5-large" (~3GB) - Good for Q&A and summarization
    - "tiiuae/falcon-rw-1b" (~2.5GB) - Lightweight but capable
    """
    print(f"Loading model: {model_name}")
    print("This may take a few minutes on first run as it downloads ~3GB...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Use float32 for CPU, float16 for GPU
        device_map="auto"  # Automatically use GPU if available
    )
    
    print(f"Model loaded successfully!")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_length=100):
    """Generate text using the loaded model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print(f"\nGenerating response to: '{prompt}'")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def main():
    """Main function to demonstrate model usage."""
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the model
    model, tokenizer = setup_model()
    
    # Example usage
    test_prompts = [
        "Artificial intelligence is",
        "The future of machine learning",
    ]
    
    for prompt in test_prompts:
        result = generate_text(model, tokenizer, prompt, max_length=100)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {result}")
        print("-" * 80)


if __name__ == "__main__":
    main()
