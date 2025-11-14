# 📜 AI Model Project – PyTorch & Hugging Face + Hardware Classifier
*A grand tome crafted in honor of thee, Almighty Bossman 👑*

---

## ⚔️ Prologue of the Arcane System  
In this sacred project, thou shalt wield the powers of PyTorch and Hugging Face, calling forth mighty transformers and vision models to classify hardware components and generate text with the wisdom of ancient neural networks.

---

## 🧙‍♂️ Chapter I – Summoning the Required Tomes  

### Prerequisites
Before thy journey begins, ensure that Python 3.8+ dwells upon thy machine.

### Install All Dependencies

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

#### Verify GPU Setup
```bash
python check_gpu.py
```

---

## 🛡️ Chapter II – The Dual Powers of This Realm

### 🎭 Power I: Text Generation with Transformers
Summon Microsoft Phi-2 for text generation and reasoning tasks.

### 🖼️ Power II: Hardware Component Classification
Train a Vision Transformer to identify computer hardware components.

---

## 🏰 Chapter III – Royal Project Structure  

```
AI Model/
├── requirements.txt              (Scroll of required incantations)
├── model_setup.py               (Text generation model)
├── train_vit_tiny.py            (Hardware classifier training)
├── test_vit_tiny.py             (Hardware classifier testing)
├── split_dataset.py             (Dataset preparation script)
├── check_gpu.py                 (GPU verification tool)
├── GPU_SETUP_COMPLETE.md        (GPU optimization guide)
├── README.md                    (This noble decree)
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

## 🦾 The Champions of Thy Realm  

### 🏰 Microsoft Phi-2 (2.7B parameters, ~3GB)  
A noble text generation warrior — strong, efficient, and well-suited for:
- Text generation  
- Question answering  
- Logical reasoning  
- General knowledge tasks  

### 👁️ Vision Transformer (ViT-Base)
A keen-eyed classifier trained to recognize:
- CPUs (Intel, AMD processors)
- GPUs (Graphics cards)
- RAM (Memory modules)
- Motherboards
- PSUs (Power supplies)

---

## 🎯 Chapter IV – The Complete Quest Workflow

### 🖼️ Quest I: Hardware Component Classification

#### Step 1: Prepare Your Image Dataset
Collect 20-50+ images for each hardware category. Place them ALL in the train folders:

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
```bash
python split_dataset.py --check
```

#### Step 3: Split Dataset (80% train, 20% validation)
```bash
python split_dataset.py --split
```

This automatically moves 20% of images to validation folders.

#### Step 4: Train the Hardware Classifier
```bash
python train_vit_tiny.py
```

**Training will show:**
- Your GPU being used (GTX 1660 Super / RTX 2070)
- Training progress and accuracy
- Validation accuracy after each epoch
- Best model saved automatically to `models/best_vit_model.pth`

**Expected Training Time:**
- 20 images/class (100 total): ~2-3 minutes on GTX 1660 Super
- 50 images/class (250 total): ~5-8 minutes
- 100 images/class (500 total): ~10-15 minutes

#### Step 5: Test Your Trained Model

**Test a single image:**
```bash
python test_vit_tiny.py --image dataset/val/cpu/test_image.jpg
```

**Test entire folder:**
```bash
python test_vit_tiny.py --directory dataset/val/gpu
```

**Interactive mode:**
```bash
python test_vit_tiny.py --interactive
```

---

### 📝 Quest II: Text Generation

#### Awaken the Text Generation Model
```bash
python model_setup.py
```

**First run warning:**  
Downloads ~3GB model (Microsoft Phi-2). May take several minutes.

#### Use in Your Own Script
```python
from model_setup import setup_model, generate_text

model, tokenizer = setup_model()
result = generate_text(model, tokenizer, "Explain what a GPU is", max_length=150)
print(result)
```

#### Summon a Different Text Model
```python
# In model_setup.py, change the model_name:
model, tokenizer = setup_model(model_name="google/flan-t5-large")
```

**Other Available Champions:**
- `google/flan-t5-large` (~3GB) - Q&A and summarization
- `tiiuae/falcon-rw-1b` (~2.5GB) - Lightweight and fast
- `stabilityai/stablelm-2-1_6b` (~3.2GB) - Modern versatile model

---

## 🏰 Chapter V – Demands of the System  

### Minimum Requirements
- **CPU:** Multi-core processor (Intel i5/Ryzen 5 or better)
- **RAM:** 8GB minimum (16GB recommended)
- **Storage:** 10GB free space
- **OS:** Windows 10/11 (with Long Paths enabled)

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

## 🛠️ Chapter VI – Remedies for Troublesome Spirits  

### ⚠️ "Could not install packages due to Long Path"
**Solution:** Run PowerShell as Admin:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```
Then **restart your computer**.

### ⚠️ "CUDA not available" (GPU not detected)
**Solution:** Reinstall PyTorch with CUDA:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Then verify:
```bash
python check_gpu.py
```

### ⚠️ Out of Memory (OOM) Error
**Solutions:**
- Reduce batch size in `train_vit_tiny.py` (change `batch_size=32` to `16` or `8`)
- Close other GPU-using programs
- Use CPU mode (slower but works)

### ⚠️ "Not enough images in dataset"
**Solution:** Need at least 2 images per class. Recommended: 20-50+ images per class.

### ⚠️ Slow Training Performance
**Check if GPU is being used:**
```bash
python check_gpu.py
```
**Optimize:**
- Ensure CUDA version matches PyTorch
- Enable mixed precision (already enabled in scripts)
- Increase batch size if you have extra VRAM

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

3. **Create applications:**
   - Web interface for hardware identification
   - Mobile app using the model
   - Automated PC builder recommendation system

4. **Combine both models:**
   - Use Vision model to identify hardware
   - Use Text model to explain specifications

---

## 📜 Quick Command Reference

```bash
# Setup & Verification
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers sentencepiece pillow
python check_gpu.py

# Dataset Management
python split_dataset.py --check
python split_dataset.py --split

# Training
python train_vit_tiny.py

# Testing
python test_vit_tiny.py --image path/to/image.jpg
python test_vit_tiny.py --directory path/to/folder
python test_vit_tiny.py --interactive

# Text Generation
python model_setup.py
```

---

## 🎓 For Your CSST 101 Final Project

This project demonstrates:
- ✅ Modern deep learning with PyTorch
- ✅ Transfer learning with pre-trained models
- ✅ Computer vision with Vision Transformers
- ✅ Natural language processing with transformers
- ✅ GPU acceleration and optimization
- ✅ Practical AI application (hardware classification)

---

*May this project serve thee well, Almighty Bossman 👑 — ruler of code, conqueror of circuits, and sovereign of machine-learning realms.*

*Forged with PyTorch 2.5.1, Transformers 4.57.1, and the power of NVIDIA Turing architecture* ⚡
