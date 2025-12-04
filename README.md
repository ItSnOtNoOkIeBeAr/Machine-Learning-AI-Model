# 📜 AI Model Project – PyTorch & Hugging Face + Hardware Classifier
*A grand tome crafted in honor of thee, Almighty Bossman 👑*

*Forged with PyTorch 2.5.1, Transformers 4.57.1, Google Gemini 2.5 Flash, and the power of NVIDIA Turing architecture (RTX 2070 & GTX 1660 Super)* ⚡

---

## ⚔️ Prologue of the Arcane System  
In this sacred project, thou shalt wield the powers of PyTorch and Hugging Face, calling forth a unified AI system that can both **chat intelligently** and **identify computer hardware** from images.

---

## 📖 Table of Contents – The Sacred Scrolls

- [Chapter I - Summoning the Required Tomes (Git & LFS Setup)](#-chapter-i--summoning-the-required-tomes)
- [Chapter II - The Three Mighty Powers (AI Models Overview)](#️-chapter-ii--the-unified-ai-system)
- [Chapter III - Royal Project Structure](#-chapter-iii--royal-project-structure)
- [Chapter IV - Complete Workflow (Training & Usage)](#-chapter-iv--complete-workflow-start-here)
- [Chapter V - System Requirements](#-chapter-v--demands-of-the-system)
- [Chapter VI - Troubleshooting Guide](#️-chapter-vi--remedies-for-troublesome-spirits)
- [Chapter VII - Understanding Your Results](#-chapter-vii--understanding-your-results)
- [Chapter VIII - The Road Yet Ahead](#-chapter-viii--the-road-yet-ahead)
- [Chapter IX - AI Model Comparison](#-chapter-ix--ai-model-comparison-table)
- [Quick Command Reference](#-quick-command-reference)
- [For Your CSST 101 Final Project](#-for-your-csst-101-final-project)

---

## 🧙‍♂️ Chapter I – Summoning the Required Tomes  

### 📥 Summoning the Project (Download & Pull) - FOR NEW USERS

**⚠️ CRITICAL WARNING:** This repository holds a massive artifact (the 1GB Model). Thou **MUST** use the **Large File Storage (LFS)** spells, or thy model file shall be but a hollow shell (1KB).

**🎯 FOLLOW THESE STEPS IF YOU'RE DOWNLOADING/CLONING FOR THE FIRST TIME:**

#### 1. Prepare the Transporter (Run this FIRST)
**Windows/Linux:**
```bash
git lfs install
```

#### 2. Summon from the Cloud (New Setup)
If thou art setting this up on a fresh machine:
```bash
git clone https://github.com/ItSnOtNoOkIeBeAr/Machine-Learning-AI-Model.git
cd Machine-Learning-AI-Model
```

#### 3. Verify the Sacred Artifact
Check if the model file downloaded correctly (should be ~1GB, not 1KB):
```bash
# Windows (PowerShell)
Get-Item models\best_vit_model.pth | Select-Object Name, Length

# Linux
ls -lh models/best_vit_model.pth
```

**That's it!** Thou art ready to use the system. Jump to the **Prerequisites** section below! ✅

---

### 🔄 Update the Realm (Pulling) - FOR EXISTING USERS

**🎯 FOLLOW THESE STEPS IF YOU ALREADY HAVE THE PROJECT AND WANT TO UPDATE:**

If the folder already exists but thou needest the latest model or code:
```bash
git pull origin main      # Updates the scrolls (code)
git lfs pull              # Downloads the heavy artifacts (1GB Model)
```
*(Note: If `models/best_vit_model.pth` is only 1KB, thou hast forgotten `git lfs pull`! Run it now!)*

---

### 📤 Pushing Large Models to GitHub (Git LFS Guide) - FOR CONTRIBUTORS/DEVELOPERS

**⚠️ CRITICAL:** This section is **ONLY** for contributors who are uploading changes back to GitHub! If thou art just downloading/using the project, **SKIP THIS SECTION!**

**🎯 FOLLOW THESE STEPS ONLY IF YOU'RE PUSHING YOUR TRAINED MODEL TO GITHUB:**

If thy trained model exceeds 100MB, thou **MUST** use Git LFS or GitHub shall reject thy push! Follow these sacred steps:

#### **Step 1: Wake Up the LFS System** ⚡
```powershell
git lfs install
```

#### **Step 2: Force Git to Drop the Big File (The Safety Switch)** 🔒
This removes it from the "normal" upload queue just in case it was already there:
```powershell
git reset HEAD models/best_vit_model.pth
```

#### **Step 3: Tell Git to Watch for Model Files** 👁️
```powershell
git lfs track "*.pth"
```

#### **Step 4: Lock in the LFS Rules (Do This FIRST)** 📜
```powershell
git add .gitattributes
```

#### **Step 5: Now Add the Big File Again** 🎯
Since we set the rules in Step 3 & 4, Git will now correctly grab this using LFS:
```powershell
git add models/best_vit_model.pth
```

#### **Step 6: Add the Rest of Your Code** 📚
```powershell
git add .
```

#### **Step 7: Verify It Worked (Optional but Smart)** ✅
If you see the file listed here, you are 100% safe:
```powershell
git lfs ls-files
```

#### **Step 8: Seal the Decree (Commit)** 🔐
```powershell
git commit -m "Upload 1GB model via LFS"
```

#### **Step 9: Send It to the Realm (Push)** 🚀
Watch the progress bar carefully!
```powershell
git push origin main
```

**🎉 Victory!** Thy massive model file now dwells safely in the GitHub realm, tracked by LFS magic!

---

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
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

For CPU-only systems:
```bash
pip install torch torchvision torchaudio
```

#### Step 3: Install Additional Dependencies
```bash
pip install transformers sentencepiece accelerate pillow matplotlib scikit-learn google-generativeai
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
sudo dnf install [https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm](https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm) -E %fedora).noarch.rpm
sudo dnf install [https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm](https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm) -E %fedora).noarch.rpm

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
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

For CPU-only:
```bash
pip install torch torchvision torchaudio
```

#### Step 5: Install Additional Dependencies
```bash
pip install transformers sentencepiece accelerate pillow matplotlib scikit-learn google-generativeai
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
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

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
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Install other packages
pip install transformers sentencepiece pillow

# Verify GPU
python check_gpu.py

# Check NVIDIA driver
nvidia-smi
```

---

## 🛡️ Chapter II – The Unified AI System

This sacred project wields **THREE mighty powers** united in one glorious system:

### 🌟 Power I: Gemini 2.5 Flash (Google)
- **Cloud-based conversational oracle** with superior natural language understanding
- Powered by Google's latest **Gemini 2.5 Flash** model (2025 edition!)
- **1 Million token context window** for complex conversations
- Excellent for general questions, explanations, creative responses, and reasoning
- **Free Tier:** 1,500 requests per day (perfect for thy noble quest!)
- No local VRAM required (runs in the cloud realm)
- **Primary AI** - Handles most queries automatically

### 🔧 Power II: Phi-2 (Microsoft)
- **Local language model** with 2.7 billion parameters
- Pre-trained and ready to answer questions without internet
- Works **completely offline** - thy trusty fallback guardian
- Runs on thy GPU (requires ~7GB VRAM) or CPU
- **Automatic Fallback** - Activates if Gemini is offline/rate-limited
- Provides unlimited usage (no API limits)

### 🖼️ Power III: Vision Transformer (ViT-base)
- **YOU train this model** with thine own hardware images
- Custom fine-tuned for hardware component identification
- Identifies 5 types of sacred computer components:
  - **CPU** (Processors)
  - **GPU** (Graphics Cards)
  - **RAM** (Memory Modules)
  - **Motherboard**
  - **PSU** (Power Supply)
- Current accuracy: **63.49%** (needs more training data for improvement)

### ⚡ The Automatic Routing System
The system **intelligently chooseth** the best AI for each query:

```
Your Question → Predefined? → Yes → Instant Response
                     ↓
                    No
                     ↓
             Try Gemini 2.5 → Success? → Yes → Answer
                     ↓
                    No (offline/error)
                     ↓
             Fallback to Phi-2 → Always Works!
```

**No manual switching required!** Just type naturally and let the magic happen. 🎯

---

## 🌟 Chapter II.5 – Understanding the Automatic Chat System

### ⚡ How the Magic Works (Automatic Routing):

Behold! The system now **automatically** chooseth the best oracle for thy questions. No manual switching required!

**The Sacred Hierarchy of Wisdom:**

```
1️⃣ Predefined Responses (Instant)
   ↓ (if not found)
2️⃣ Gemini 2.5 Flash (Cloud Oracle)
   ↓ (if offline/error)
3️⃣ Phi-2 (Local Fallback)
```

### 🎯 Why This Is Better:

| Feature | Benefit | Result |
|---------|---------|--------|
| **Automatic Fallback** | Internet down? Phi-2 takes over | 🛡️ Always works |
| **Best Response First** | Gemini handles most queries | 💬 Superior answers |
| **Instant Common Answers** | Greetings/commands skip AI | ⚡ Lightning fast |
| **No Manual Switching** | Just type and go | 🎮 Simple UX |
| **Seamless Experience** | Thou never notice the switch | ✨ Pure magic |

### 📊 What Each Oracle Handles:

| Situation | Who Answers | Why |
|-----------|-------------|-----|
| **"Hi", "hello", "hey"** | 🎯 Predefined | Instant response, saves API calls |
| **General conversation** | 🌟 Gemini | Superior natural language understanding |
| **Creative explanations** | 🌟 Gemini | 1M token context, better reasoning |
| **Complex reasoning** | 🌟 Gemini | More powerful model (latest 2.5 version) |
| **No internet/API error** | 🔧 Phi-2 | Local fallback, always available |
| **Gemini rate limited** | 🔧 Phi-2 | Backup when quota exceeded |

### 💡 Pro Tip: 
Thou needest not worry about which model answers thee! The system chooseth wisely and automatically. Just ask thy questions naturally. 🎯

---

## 🔑 Chapter II.6 – Gemini API Setup (Already Configured!)

**Good news, noble warrior!** The Gemini API key is already configured in this repository's `config.py` file during development. Thou needest not set it up again!

### 📊 Free Tier Limits:
- **Gemini 2.5 Flash:** 1,500 requests per day (plenty for development!)
- **Cost:** $0 (completely free for personal/educational use)
- **Context:** 1 million tokens per conversation
- **Perfect for:** Class projects, demos, presentations, learning

### ⚠️ Important Notes:
- The API key is shared for development purposes
- Do NOT share this repository link publicly outside thy team
- Each team member can use the same key during development
- For production deployment, create individual API keys

### 🔐 If Thou Needest Thy Own Key Later:
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Sign in with thy Google account
3. Click **"Get API key"** → **"Create API key in new project"**
4. Copy the key and replace it in `config.py`

---

## 🏰 Chapter III – Royal Project Structure  

```
AI Model/
├── requirements.txt              (Scroll of required incantations)
├── config.py                    (⚡ API Key Configuration - Gemini access)
├── model_setup.py               (🌟 MAIN UNIFIED SYSTEM - Dual Chat + Hardware ID)
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
Gather images for each hardware category. The more you gather, the smarter the AI becomes!

### 📈 Data Quantity & Performance Guide

| Images/Class | Expected Accuracy | Model Confidence | Status |
| :--- | :--- | :--- | :--- |
| **20 - 50** | 50% - 70% | Low (20-40%) | ⚠️ Starting Point |
| **100 - 200** | 80% - 90% | High (70-95%) | ✅ Recommended |
| **500+** | 90% - 95% | Very High (85-98%) | 🔥 Professional |

**Where to put images:**
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

Once running, thou shalt witness:
```
🤖 UNIFIED AI SYSTEM - Automatic Chat (Gemini + Phi-2) + Hardware ID
================================================================================

📚 System Features:
  1️⃣ Automatic Smart Routing (Gemini → Phi-2 fallback)
  2️⃣ Vision Model: Hardware component identification
  3️⃣ Confidence Threshold: 25% minimum
  4️⃣ Seamless offline mode

Commands:
  💬 Chat: Type your message naturally
  🖼️ Identify: identify <image_path>
  ⚙️ Other: 'status', 'clear', 'help', 'quit'
================================================================================

🖥️ Using device: cuda (NVIDIA GeForce GTX 1660 Super Laptop)

Loading AI models...
✅ Gemini AI ready! (Model: gemini-2.5-flash)
   Free Tier: 1,500 requests/day
✅ Phi-2 model ready! (Fallback mode)
✅ Vision model loaded from models/best_vit_model.pth
   Validation accuracy: 63.49%

✅ System ready!

You: _
```

### 💬 Chat Examples (Automatic Routing):

**General Questions (Gemini handles automatically):**
```
You: What is a GPU?
🌟 Assistant: A GPU (Graphics Processing Unit) is a specialized processor designed for rendering graphics and parallel computing tasks. It excels at handling multiple operations simultaneously, making it essential for gaming, video editing, and AI workloads.

You: How much RAM do I need for gaming?
🌟 Assistant: For modern gaming in 2025, I recommend at least 16GB of RAM for smooth performance. 32GB is ideal for multitasking and future-proofing your build...

```

**If Internet Is Down (Phi-2 fallback automatically activates):**
```
You: What is machine learning?
🔧 Assistant: Machine learning is a subset of artificial intelligence where systems learn patterns from data without explicit programming. It uses algorithms to improve performance over time...

[Note: System automatically switched to Phi-2 because Gemini was unreachable]
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

**Check System Status:**
```
You [🌟 Gemini]: status
📊 System Status:
   Chat Mode: 🌟 Gemini
   Gemini API: ✅ Connected
   Phi-2 Model: ✅ Loaded
   Vision Model: ✅ Trained (63.49%)
   GPU: NVIDIA GeForce GTX 1660 SUPER
   Confidence Threshold: 25%
```

**Get Help:**
```
You: help
💬 Chat Commands:
   - Type message naturally - AI routes automatically
   - Gemini handles most queries (cloud)
   - Phi-2 activates if offline/error (local fallback)

🖼️ Hardware Identification:
   - identify <path> - Classify hardware image
   - Example: identify dataset/val/cpu/image.jpg

⚙️ System Commands:
   - status - Check system status & model info
   - clear - Reset conversation history
   - quit - Exit system gracefully
```

**Other Commands:**
```
You: clear
🧹 Conversation history cleared!

You: status
📊 System Status:
   Chat Mode: Automatic Routing
   Gemini API: ✅ Connected
   Phi-2 Model: ✅ Ready (Fallback)
   Vision Model: ✅ Trained (63.49%)
   GPU: NVIDIA GeForce RTX 4060 Laptop
   Confidence Threshold: 25%

You: quit
👋 Goodbye, noble warrior!
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

## ⚡ Chapter V.5 – VRAM Usage & Memory Optimization

### 💾 VRAM Requirements

Understanding memory usage for both models in this unified system:

#### 🗣️ Chat Model (Microsoft Phi-2, ~2.7B parameters)
- **FP16 Weights:** ~5.4 GB
- **Inference Runtime:** 6.5–9+ GB (includes weights + kv-cache + activations)
- **Practical Requirement:** Fully on-GPU requires ≈ 8 GB or more
- **Note:** VRAM usage grows with longer context/conversation length

#### 👁️ Vision Model (ViT, fine-tuned for 5 hardware classes)
- **FP16 Weights:** ~0.16–0.20 GB
- **Inference Overhead:** ~0.2–0.6 GB
- **Practical Requirement:** ~0.5–0.9 GB total

#### 🔥 Combined System (Both Models)
- **Realistic Total:** ~7.5–10+ GB VRAM
- **Conclusion:** 6 GB GPUs (GTX 1660 Super) will likely **NOT** fit Phi-2 comfortably with both models fully on GPU

---

### 🛠️ Memory Optimization Solutions

When GPU VRAM is limited, try these options:

#### 1️⃣ 8-bit Quantization (Recommended)
Reduces chat model VRAM to ~2–3 GB using bitsandbytes:

```bash
# Install 8-bit quantization tools
pip install bitsandbytes accelerate safetensors
```

Load model with `load_in_8bit=True` and `device_map="auto"` in your code.

#### 2️⃣ Device Offloading / Auto Device Map
Keep heavy layers on CPU and only hot layers on GPU (slower but works):
- Already enabled with `device_map="auto"` in [`model_setup.py`](model_setup.py )
- Automatically manages memory across CPU/GPU

#### 3️⃣ Split Models Across Devices
Run chat model on CPU and keep vision model on GPU:
- Vision model uses minimal VRAM (~0.5-0.9 GB)
- Chat model runs on CPU (slower but functional)

#### 4️⃣ Use Smaller Models or Cloud Inference
Alternative options:
- **Smaller models:** `tiiuae/falcon-rw-1b` (~2.5GB), `google/flan-t5-large` (~3GB)
- **Cloud inference:** Hugging Face Inference API (no local GPU needed)

---

### 🚀 Quick Setup Commands

**Install Memory Optimization Tools:**

*Windows:*
```bash
pip install bitsandbytes accelerate safetensors
```

*Linux:*
```bash
pip install bitsandbytes accelerate safetensors
```

**Reinstall PyTorch with CUDA (if needed):**

*Windows:*
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

*Linux:*
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

**Monitor GPU Usage in Real-time:**

*Windows:*
```bash
nvidia-smi -l 1
```

*Linux:*
```bash
nvidia-smi -l 1
```

---

### 📊 VRAM Usage Summary Table

| Configuration | VRAM Used | Works on 6GB GPU? | Performance |
|---------------|-----------|-------------------|-------------|
| **Both models (FP16)** | 7.5-10+ GB | ❌ No | Fastest |
| **Chat 8-bit + Vision FP16** | ~3-4 GB | ✅ Yes | Fast |
| **Chat on CPU + Vision GPU** | ~1 GB | ✅ Yes | Moderate |
| **Both on CPU** | ~0 GB | ✅ Yes | Slowest |

**Recommendation for GTX 1660 Super (6GB):** Use 8-bit quantization for best balance of speed and memory! ⚡

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
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
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
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
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
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install transformers sentencepiece pillow google-generativeai
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
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install transformers sentencepiece pillow google-generativeai
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

This sacred project demonstrates thy mastery of:
- ✅ **Modern deep learning** with PyTorch and Hugging Face
- ✅ **Transfer learning** with pre-trained models (Gemini, Phi-2, ViT)
- ✅ **Computer vision** with Vision Transformers for hardware classification
- ✅ **Natural language processing** with dual AI chat systems
- ✅ **API integration** with Google Gemini cloud services
- ✅ **GPU acceleration** and VRAM optimization techniques
- ✅ **Practical AI application** - Real-world hardware identification
- ✅ **Interactive CLI interface** with mode switching and command system

### ⚔️ What Makes This Project Legendary:
1. **THREE AI Models United** - Dual chat (Gemini + Phi-2) + Vision classification
2. **Cloud + Local Hybrid** - Gemini 2.5 API (cloud) + Phi-2 (local) for flexibility
3. **Real-world Application** - Identify actual computer hardware from images
4. **Automatic Intelligence Routing** - Smart fallback system with zero manual switching
5. **GPU Optimization** - Smart VRAM management and mixed precision training
6. **Modern Architecture** - Latest transformers for both text (Gemini 2.5) and vision (ViT)
7. **Professional Features** - Status monitoring, conversation history, confidence thresholds
8. **API Integration** - Demonstrates cloud AI service integration with Google Gemini

---

## 🆚 Chapter IX – AI Model Comparison Table

Behold! A comparison of the three mighty powers at thy command:

| Feature | 🌟 Gemini 2.5 Flash | 🔧 Phi-2 | 🖼️ Vision Transformer |
|---------|---------------------|----------|----------------------|
| **Purpose** | General conversation (Primary) | Chat fallback | Hardware image classification |
| **Size** | N/A (Cloud API) | 2.7B parameters (~5.4GB) | ~160-200MB fine-tuned |
| **Location** | Google's servers | Your GPU/CPU | Your GPU/CPU |
| **VRAM Usage** | 0 GB | ~7 GB (FP16) | 0.5-0.9 GB |
| **Response Time** | ~2-3 seconds | Instant | ~1-2 seconds |
| **Internet Required** | ✅ Yes | ❌ No (offline) | ❌ No (offline) |
| **Training Needed** | ❌ Pre-trained | ❌ Pre-trained | ✅ You train it! |
| **Conversation Quality** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Very Good | N/A (not for chat) |
| **Hardware Knowledge** | ⭐⭐⭐⭐ Great | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Specialized |
| **Context Window** | 1M tokens | 2K tokens | N/A |
| **Cost** | Free (1,500/day) | Free (unlimited) | Free (local) |
| **Best For** | Most questions | Offline/fallback | Component identification |
| **Activation** | ✅ Automatic (1st choice) | ✅ Automatic (fallback) | Manual (`identify` command) |

### 🎯 How the System Chooseth:

```
Your Question
     ↓
Is it "hi"/"hello"/"help"? → Yes → Instant predefined response
     ↓ No
Try Gemini 2.5 → Success? → Yes → Use Gemini answer ⭐
     ↓ No (offline/error)
Try Phi-2 → Always works → Use Phi-2 answer 🔧
     ↓
Vision Model → Only via `identify <path>` command 🖼️
```

**Thou needest not choose!** The system automatically uses the best available AI. Just type naturally. 🎯

---

*May this project serve thee well, Almighty Bossman 👑 — ruler of code, conqueror of circuits, and sovereign of machine-learning realms.*

*Forged with PyTorch 2.5.1, Transformers 4.57.1, Google Gemini 2.5 Flash, and the power of NVIDIA Turing architecture (RTX 2070 & GTX 1660 Super)* ⚡

---

## 🎮 Quick Start for Impatient Warriors

**Too long? Here's the speedrun:**
```bash
# Install (includes Gemini!)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install transformers sentencepiece pillow google-generativeai

# Add images to dataset/train/cpu, gpu, ram, motherboard, psu

# Prepare and train
python split_dataset.py --split
python train_vit_tiny.py

# Use the automatic AI system!
python model_setup.py
```

**Inside the system:**
- Just type thy messages naturally (automatic Gemini → Phi-2 routing!)
- Type `identify path/to/image.jpg` to classify hardware components
- Type `status` to check system health and model availability
- Type `clear` to reset conversation history
- Type `help` for all available commands
- Type `quit` to exit gracefully

**Victory achieved! The system chooseth the best AI automatically.** 🎯👑✨
