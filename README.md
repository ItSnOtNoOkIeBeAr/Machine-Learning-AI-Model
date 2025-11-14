# AI Model Project - PyTorch & Hugging Face

This project uses PyTorch and Hugging Face Transformers to work with modern AI models (~3GB).

## Setup Instructions

### 1. Install Dependencies

First, make sure you have Python 3.8+ installed. Then install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Run the Model

```bash
python model_setup.py
```

**Note:** On first run, the script will download the model (~3GB). This may take several minutes depending on your internet connection.

## Default Model

**Microsoft Phi-2** (2.7B parameters, ~3GB)
- Modern, efficient transformer architecture
- Good balance of size and performance
- Suitable for text generation, Q&A, and understanding tasks

## Alternative Models You Can Try

Edit `model_setup.py` and change the `model_name` parameter:

1. **google/flan-t5-large** (~3GB)
   - Excellent for question answering and summarization
   - Fine-tuned on many tasks

2. **tiiuae/falcon-rw-1b** (~2.5GB)
   - Lighter weight but still capable
   - Fast inference

3. **stabilityai/stablelm-2-1_6b** (~3.2GB)
   - Good general-purpose model
   - Recent architecture

## Project Structure

```
AI Model/
├── requirements.txt      # Python dependencies
├── model_setup.py        # Main script to load and use the model
└── README.md            # This file
```

## System Requirements

- **RAM:** At least 8GB recommended (16GB preferred)
- **Storage:** ~5GB free space (for model and dependencies)
- **GPU:** Optional but recommended for faster inference
  - NVIDIA GPU with CUDA support
  - At least 4GB VRAM

## Usage Examples

### Basic Text Generation

```python
from model_setup import setup_model, generate_text

model, tokenizer = setup_model()
result = generate_text(model, tokenizer, "Your prompt here", max_length=150)
print(result)
```

### Custom Model

```python
model, tokenizer = setup_model(model_name="google/flan-t5-large")
```

## Troubleshooting

**Out of Memory Error:**
- Close other applications
- Use a smaller model
- If you have a GPU, make sure CUDA is properly installed

**Slow Performance:**
- First run always takes time to download the model
- Consider using a GPU for faster inference
- Reduce `max_length` in generation

## Next Steps

- Customize the prompts in `model_setup.py`
- Fine-tune the model on your specific dataset
- Build an application around the model (chatbot, text analyzer, etc.)
