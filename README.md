# 📜 AI Model Project – PyTorch & Hugging Face + Hardware Classifier
*A grand tome crafted in honor of thee, Almighty Bossman 👑*

---

## ⚔️ Prologue of the Arcane System  
In this sacred project, thou shalt wield the powers of PyTorch and Hugging Face, calling forth a unified AI system that can both **chat intelligently** and **identify computer hardware** from images.

---

## 🧙‍♂️ Chapter I – Summoning the Required Tomes  

### Prerequisites
Before thy journey begins, ensure that Python 3.8+ dwells upon thy machine.

---

### 🪟 Windows Installation

#### Step 1: Enable Windows Long Paths (One-time setup)
Open PowerShell as **Administrator** and run:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```
**⚠️ RESTART YOUR COMPUTER after this step!**

#### Step 2: Install PyTorch with CUDA (for GPU acceleration)
For NVIDIA GPUs (GTX 1660 Super, RTX 2070, etc.):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only systems:
```bash
pip install torch torchvision torchaudio
```

#### Step 3: Install Additional Dependencies
```bash
pip install transformers sentencepiece accelerate pillow matplotlib scikit-learn
```

#### Step 4: Verify GPU Setup
```bash
python check_gpu.py
```

---

### 🐧 Linux Installation (Ubuntu/Debian/Mint/Arch/Fedora)

#### Step 1: Update System and Install Python
**Ubuntu/Debian/Mint:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv git
```

**Arch Linux:**
```bash
sudo pacman -Syu
sudo pacman -S python python-pip git
```

**Fedora:**
```bash
sudo dnf update
sudo dnf install python3 python3-pip git
```

#### Step 2: Install NVIDIA Drivers (for GPU acceleration)
**Ubuntu/Debian/Mint:**
```bash
# Check if you have NVIDIA GPU
lspci | grep -i nvidia

# Install NVIDIA drivers
sudo apt install nvidia-driver-535

# Reboot after installation
sudo reboot
```

**Arch Linux:**
```bash
# Install NVIDIA drivers
sudo pacman -S nvidia nvidia-utils

# Reboot
sudo reboot
```

**Fedora:**
```bash
# Enable RPM Fusion repositories
sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm

# Install NVIDIA drivers
sudo dnf install akmod-nvidia
sudo reboot
```

#### Step 3: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv ai_env

# Activate it
source ai_env/bin/activate

# Your terminal should now show (ai_env)
```

#### Step 4: Install PyTorch with CUDA
For NVIDIA GPUs:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only:
```bash
pip install torch torchvision torchaudio
```

#### Step 5: Install Additional Dependencies
```bash
pip install transformers sentencepiece accelerate pillow matplotlib scikit-learn
```

#### Step 6: Verify GPU Setup
```bash
python check_gpu.py
```

#### Step 7: Check CUDA and GPU
```bash
# Check NVIDIA driver installation
nvidia-smi

# Should show your GPU (GTX 1660 Super, RTX 2070, etc.)
```

---

### 🎯 Quick Start Commands by OS

**Windows:**
```bash
# Install PyTorch + CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other packages
pip install transformers sentencepiece pillow

# Verify GPU
python check_gpu.py
```

**Linux:**
```bash
# Create and activate virtual environment
python3 -m venv ai_env
source ai_env/bin/activate

# Install PyTorch + CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other packages
pip install transformers sentencepiece pillow

# Verify GPU
python check_gpu.py

# Check NVIDIA driver
nvidia-smi
```

---

## 🛡️ Chapter II – The Unified AI System

This project combines **two powerful AI models** into one unified system:

### 🎭 Power I: Chat AI (Microsoft Phi-2)
- Pre-trained language model (2.7B parameters)
- Answer questions about hardware, technology, and general topics
- Explain technical concepts
- Have natural conversations

### 🖼️ Power II: Hardware Classifier (Vision Transformer)
- YOU train this model with your hardware images
- Identifies 5 types of computer components:
  - **CPU** (Processors)
  - **GPU** (Graphics Cards)
  - **RAM** (Memory Modules)
  - **Motherboard**
  - **PSU** (Power Supply)

---

## 🏰 Chapter III – Royal Project Structure  

```
AI Model/
├── requirements.txt              (Scroll of required incantations)
├── model_setup.py               (🌟 MAIN UNIFIED SYSTEM - Chat + Hardware ID)
├── train_vit_tiny.py            (Hardware classifier training)
├── test_vit_tiny.py             (Hardware classifier testing - standalone)
├── split_dataset.py             (Dataset preparation script)
├── check_gpu.py                 (GPU verification tool)
├── GPU_SETUP_COMPLETE.md        (GPU optimization guide)
├── README.md                    (This noble decree)
│
├── models/
│   └── best_vit_model.pth       (Your trained vision model - created after training)
│
└── dataset/
    ├── train/                   (Training images - 80%)
    │   ├── cpu/
    │   ├── gpu/
    │   ├── ram/
    │   ├── motherboard/
    │   └── psu/
    └── val/                     (Validation images - 20%)
        ├── cpu/
        ├── gpu/
        ├── ram/
        ├── motherboard/
        └── psu/
```

---

## 🎯 Chapter IV – Complete Workflow (Start Here!)

### 📋 Phase 1: Prepare Your Dataset (Required for Hardware ID)

#### Step 1: Collect Hardware Images
Gather 20-50+ images for each hardware category:

```bash
dataset/train/cpu/           ← Add CPU images here
dataset/train/gpu/           ← Add GPU images here
dataset/train/ram/           ← Add RAM images here
dataset/train/motherboard/   ← Add motherboard images here
dataset/train/psu/           ← Add PSU images here
```

**Image Sources:**
- Google Images
- Amazon/Newegg product photos
- Manufacturer websites (Intel, AMD, NVIDIA, Corsair)
- Your own hardware photos

#### Step 2: Check Your Dataset

**Windows:**
```bash
python split_dataset.py --check
```

**Linux:**
```bash
python3 split_dataset.py --check
```

#### Step 3: Split Dataset (80% train, 20% validation)

**Windows:**
```bash
python split_dataset.py --split
```

**Linux:**
```bash
python3 split_dataset.py --split
```

---

### 🎓 Phase 2: Train the Vision Model

**Windows:**
```bash
python train_vit_tiny.py
```

**Linux:**
```bash
python3 train_vit_tiny.py
```

**What happens during training:**
- Uses your GTX 1660 Super / RTX 2070 GPU automatically
- Shows training progress and accuracy
- Validates after each epoch
- Saves best model to `models/best_vit_model.pth`

**Expected Training Time:**
- 20 images/class (100 total): ~2-3 minutes
- 50 images/class (250 total): ~5-8 minutes
- 100 images/class (500 total): ~10-15 minutes

---

### 🚀 Phase 3: Use the Unified AI System

## 🌟 **MAIN COMMAND - Start the Unified System**

**Windows:**
```bash
python model_setup.py
```

**Linux:**
```bash
python3 model_setup.py
```

**First run:** Downloads Microsoft Phi-2 model (~3GB). May take several minutes.

---

## 💬 Using the Unified System

Once running, you'll see:
```
🤖 UNIFIED AI SYSTEM - Chat + Hardware Identification
================================================================================

📚 How This Works:
  1️⃣ Chat Model (Phi-2): Already trained, ready to chat
  2️⃣ Vision Model: YOU trained this with hardware images

Commands:
  💬 Chat: Type your message
  🖼️ Identify: identify <image_path>
  ⚙️ Other: 'quit', 'clear', 'help', 'status'
================================================================================

🖥️ Using device: cuda (NVIDIA GeForce GTX 1660 SUPER)

Loading AI models...
✅ System ready!

You: _
```

### 💬 Chat Examples:

```
You: What is a GPU?
🤖 Assistant: A GPU (Graphics Processing Unit) is a specialized processor designed for rendering graphics and parallel computing tasks...

You: How much RAM do I need for gaming?
🤖 Assistant: For modern gaming in 2024, I recommend at least 16GB of RAM...

You: Explain machine learning
🤖 Assistant: Machine learning is a subset of artificial intelligence...
```

### 🖼️ Hardware Identification Examples:

```
You: identify dataset/val/cpu/intel_i9.jpg

🔍 Analyzing: dataset/val/cpu/intel_i9.jpg
⏳ Processing...

🎯 Prediction: CPU
📊 Confidence: 96.78%

📈 Top 3 Predictions:
   1. CPU: 96.78%
   2. MOTHERBOARD: 2.15%
   3. GPU: 1.07%

🤖 AI Explanation:
   A CPU (Central Processing Unit) is the primary processor that executes 
   instructions and performs calculations. It acts as the brain of the 
   computer system.
```

**More identification examples:**
```
You: identify C:\Users\Matthew Dee\Pictures\my_gpu.jpg
You: identify dataset/train/ram/corsair_vengeance.png
You: identify D:\Downloads\hardware_photo.jpg
```

### ⚙️ System Commands:

```
You: status
📊 System Status:
   Chat Model: ✅ Ready (Pre-trained Phi-2)
   Vision Model: ✅ Trained
   GPU: NVIDIA GeForce GTX 1660 SUPER

You: help
💬 Chat Commands:
   - Type message to chat with AI

🖼️ Hardware Identification:
   - identify <path> - Classify hardware image
   - Example: identify dataset/val/cpu/image.jpg

⚙️ System Commands:
   - status - Check system status
   - clear - Reset conversation
   - quit - Exit system

You: clear
🧹 Conversation history cleared!

You: quit
👋 Goodbye!
```

---

## 🧪 Optional: Test Vision Model Separately

If you want to test the vision model without the chat interface:

**Test a single image:**

*Windows:*
```bash
python test_vit_tiny.py --image dataset/val/cpu/test_image.jpg
```

*Linux:*
```bash
python3 test_vit_tiny.py --image dataset/val/cpu/test_image.jpg
```

**Test entire folder:**

*Windows:*
```bash
python test_vit_tiny.py --directory dataset/val/gpu
```

*Linux:*
```bash
python3 test_vit_tiny.py --directory dataset/val/gpu
```

**Interactive mode:**

*Windows:*
```bash
python test_vit_tiny.py --interactive
```

*Linux:*
```bash
python3 test_vit_tiny.py --interactive
```

---

## 🏰 Chapter V – Demands of the System  

### Minimum Requirements
- **CPU:** Multi-core processor (Intel i5/Ryzen 5 or better)
- **RAM:** 8GB minimum (16GB recommended)
- **Storage:** 10GB free space
- **OS:** Windows 10/11 or Linux

### Recommended for GPU Training
- **GPU:** NVIDIA GTX 1660 Super or RTX 2070 (or better)
- **VRAM:** 4-6GB minimum
- **CUDA:** Automatically installed with PyTorch
- **Architecture:** Turing or newer (supports FP16 mixed precision)

### GPU Performance Comparison
- **GTX 1660 Super** (6GB VRAM): ~2-3x faster than CPU
- **RTX 2070** (8GB VRAM): ~30-40% faster than GTX 1660 Super (has Tensor Cores)
- Both excellent for this project! 🔥

---

## ⚡ VRAM Usage Estimates & Recommendations

Short practical estimates for the models used in this project.

- Chat model — Microsoft Phi‑2 (~2.7B parameters)
  - FP16 weights ≈ 5.4 GB
  - Inference runtime (weights + kv‑cache + activations) ≈ 6.5–9+ GB
  - Practical: fully on‑GPU requires ≈ 8 GB or more; grows with context length

- Vision model — ViT (fine‑tuned for 5 classes)
  - FP16 weights ≈ 0.16–0.20 GB
  - Inference overhead ≈ 0.2–0.6 GB
  - Practical: ≈ 0.5–0.9 GB

- Combined (both resident on GPU)
  - Realistic total ≈ 7.5–10+ GB
  - Conclusion: 6 GB GPUs (GTX 1660 Super) will likely NOT fit Phi‑2 comfortably if both are fully on GPU

Recommended options when GPU VRAM is limited:
1. 8‑bit quantization (bitsandbytes): reduces chat model VRAM to ~2–3 GB.
   - Install: `pip install bitsandbytes accelerate safetensors`
   - Load with `load_in_8bit=True` and `device_map="auto"`.
2. Device offloading / automatic device map: keep parts on CPU and only use GPU for hot layers.
3. Run chat model on CPU and keep vision model on GPU (vision uses little VRAM).
4. Use a smaller chat model or hosted inference (Hugging Face Inference API) if local resources are insufficient.

Quick commands:
```bash
# Install 8-bit tooling
pip install bitsandbytes accelerate safetensors

# Example: reinstall PyTorch with CUDA if needed (Windows)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Add these options to your workflow based on available VRAM.  

---

## 🛠️ Chapter VI – Remedies for Troublesome Spirits  

### ⚠️ "Could not install packages due to Long Path" (Windows)
**Solution:** Run PowerShell as Admin:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```
Then **restart your computer**.

### ⚠️ "CUDA not available" (GPU not detected)

**Windows Solution:**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python check_gpu.py
```

**Linux Solution:**
```bash
# Check NVIDIA driver first
nvidia-smi

# If driver not found, install it (Ubuntu/Debian/Mint)
sudo apt install nvidia-driver-535
sudo reboot

# Then reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python3 check_gpu.py
```

### ⚠️ "Vision model not trained yet" message
**Solution:** You need to train the vision model first:
```bash
# Windows
python train_vit_tiny.py

# Linux
python3 train_vit_tiny.py
```

The chat will still work, but hardware identification won't work until trained.

### ⚠️ Out of Memory (OOM) Error
**Solutions:**
- Reduce batch size in `train_vit_tiny.py` (change `batch_size=32` to `16` or `8`)
- Close other GPU-using programs
- Use CPU mode (slower but works)

### ⚠️ "Not enough images in dataset"
**Solution:** Need at least 2 images per class. Recommended: 20-50+ images per class.

### ⚠️ Slow Training Performance

**Check if GPU is being used:**

*Windows:*
```bash
python check_gpu.py
```

*Linux:*
```bash
python3 check_gpu.py
# Also check GPU utilization in real-time
nvidia-smi -l 1
```

**Optimize:**
- Ensure CUDA version matches PyTorch
- Enable mixed precision (already enabled in scripts)
- Increase batch size if you have extra VRAM
- On Linux: Check if GPU is not being used by another process with `nvidia-smi`

---

## 📊 Chapter VII – Understanding Your Results

### Training Metrics
- **Training Accuracy:** How well model learns from training data
- **Validation Accuracy:** True performance on unseen data (most important!)
- **Loss:** Lower is better (measures prediction errors)

### Good Results Indicators
- Validation accuracy > 80% = Good model
- Validation accuracy > 90% = Excellent model
- Training and validation accuracy close = No overfitting ✅
- Training much higher than validation = Overfitting ⚠️ (need more data)

---

## 🚀 Chapter VIII – The Road Yet Ahead  

### Expand Your Powers
1. **Add more hardware categories:**
   - SSDs, Hard Drives, Cooling systems, Cases, etc.
   
2. **Improve accuracy:**
   - Collect 100+ images per category
   - Use data augmentation
   - Train for more epochs

3. **Deploy your system:**
   - Create web interface with Streamlit
   - Build Discord bot
   - Make mobile app
   - Host on cloud server

4. **Advanced features:**
   - Multi-language support
   - Voice chat integration
   - Batch image processing
   - Hardware recommendation system

---

## 📜 Quick Command Reference

### 🎯 Main Commands (What You'll Use Most)

**🪟 Windows:**
```bash
# 1. Setup (one-time)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers sentencepiece pillow
python check_gpu.py

# 2. Prepare dataset
python split_dataset.py --split

# 3. Train vision model
python train_vit_tiny.py

# 4. 🌟 USE THE UNIFIED SYSTEM 🌟
python model_setup.py
```

**🐧 Linux:**
```bash
# 1. Setup (one-time)
python3 -m venv ai_env
source ai_env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers sentencepiece pillow
python3 check_gpu.py

# 2. Prepare dataset
python3 split_dataset.py --split

# 3. Train vision model
python3 train_vit_tiny.py

# 4. 🌟 USE THE UNIFIED SYSTEM 🌟
python3 model_setup.py
```

### 📋 Complete Workflow Summary

```
Step 1: Install dependencies ✅
Step 2: Add 20+ images per hardware category to dataset/train/ ✅
Step 3: python split_dataset.py --split ✅
Step 4: python train_vit_tiny.py ✅
Step 5: python model_setup.py ✅ ← START USING YOUR AI!
```

---

## 🎓 For Your CSST 101 Final Project

This project demonstrates:
- ✅ Modern deep learning with PyTorch
- ✅ Transfer learning with pre-trained models
- ✅ Computer vision with Vision Transformers
- ✅ Natural language processing with transformers
- ✅ GPU acceleration and optimization
- ✅ Practical AI application (unified chat + image classification)
- ✅ Interactive command-line interface

### What Makes This Project Special:
1. **Two AI Models in One System** - Chat and Vision combined
2. **Real-world Application** - Identify actual computer hardware
3. **GPU Optimization** - Uses CUDA acceleration
4. **Modern Architecture** - Transformers for both text and vision
5. **Interactive Experience** - Natural conversation + image analysis

---

*May this project serve thee well, Almighty Bossman 👑 — ruler of code, conqueror of circuits, and sovereign of machine-learning realms.*

*Forged with PyTorch 2.5.1, Transformers 4.57.1, and the power of NVIDIA Turing architecture* ⚡

---

## 🎮 Quick Start for Impatient Warriors

**Too long? Here's the speedrun:**
```bash
# Install
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers sentencepiece pillow

# Add images to dataset/train/cpu, gpu, ram, motherboard, psu

# Prepare and train
python split_dataset.py --split
python train_vit_tiny.py

# Use the system!
python model_setup.py
```

Type `help` once inside for commands. Type `identify path/to/image.jpg` to identify hardware. Chat normally for questions. GG! 🎯
