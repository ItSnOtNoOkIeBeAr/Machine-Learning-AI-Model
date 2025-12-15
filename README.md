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
- [Chapter IX - The Sacred Tech Stack (What Powers This Realm)](#-chapter-ix--the-sacred-tech-stack)
- [Chapter X - AI Model Comparison](#-chapter-x--ai-model-comparison-table)
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

**⚠️ Alternative: If thy model is ignored by .gitignore, force-add it thus:**
```powershell
git add -f models/best_vit_model.pth
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
For thy trained model with excellent accuracy, proclaim thus:
```powershell
git commit -m "Add trained vision model (82.50% accuracy)"
```

Or for general large files:
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
- Current accuracy: **82.50%** ✅ (Production-ready! Excellent performance)
- Training results: 100% train accuracy, 82.50% validation accuracy (20 epochs)

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
├── api_server.py                (🌐 FastAPI Backend - REST API for web integration)
├── train_vit_tiny.py            (Hardware classifier training)
├── test_vit_tiny.py             (Hardware classifier testing - standalone)
├── split_dataset.py             (Dataset preparation script)
├── check_gpu.py                 (GPU verification tool)
├── start_tunnel.bat             (⚡ Quick Cloudflare tunnel launcher)
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

🖥️ Using device: cuda (NVIDIA GeForce GTX 1660 Super)

Loading AI models...
✅ Gemini AI ready! (Model: gemini-2.5-flash)
   Free Tier: 1,500 requests/day
✅ Phi-2 model ready! (Fallback mode)
✅ Vision model loaded from models/best_vit_model.pth
   Validation accuracy: 82.50% ✅

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
You: identify dataset/val/ram/corsair_ram.jpg

🔍 Analyzing: dataset/val/ram/corsair_ram.jpg
⏳ Processing...

🎯 Prediction: RAM
📊 Confidence: 38.33%

📈 Top 3 Predictions:
   1. RAM: 38.33%
   2. MOTHERBOARD: 18.85%
   3. CPU: 18.62%

🤖 AI Explanation:
   RAM (Random Access Memory) provides temporary storage for active data
   and programs. More RAM allows better multitasking and faster performance.
```

**Another example with high confidence:**
```
You: identify dataset/val/cpu/intel_i9.jpg

🔍 Analyzing: dataset/val/cpu/intel_i9.jpg
⏳ Processing...

🎯 Prediction: CPU
📊 Confidence: 89.45%

📈 Top 3 Predictions:
   1. CPU: 89.45%
   2. MOTHERBOARD: 7.32%
   3. GPU: 2.11%

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
   Chat Mode: Automatic Routing
   Gemini API: ✅ Connected
   Phi-2 Model: ✅ Ready (Fallback)
   Vision Model: ✅ Trained (82.50% accuracy)
   GPU: NVIDIA GeForce RTX 2070 / GTX 1660 Super
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
   Vision Model: ✅ Trained (82.50% accuracy)
   GPU: NVIDIA GeForce RTX 2070 / GTX 1660 Super
   Confidence Threshold: 25%

You: quit
👋 Goodbye, noble warrior!
```

---

## 🌐 Hosting Thy AI Models Online (Cloudflare Tunnel)

**Wish to share thy mighty AI with the realm?** Follow these sacred steps to expose thy models via the internet!

### 🎯 **What Thou Needest:**
- ✅ Thy PC running (with the sacred models loaded)
- ✅ Internet connection (for the cloud tunnel)
- ✅ Two PowerShell windows (one for API, one for tunnel)

---

### **📋 Step 1: Start the API Server** 🚀

Open **PowerShell Terminal 1** and proclaim:

```powershell
cd "C:\Users\Matthew Dee\Documents\School work\LSPU\School works\3rd year\1st sem\CSST 101\Final Project\AI Model"

python api_server.py
```

**Wait for the prophecy:**
```
================================================================================
✅ All models loaded successfully!
📡 API ready at http://localhost:8000
📖 Documentation at http://localhost:8000/docs
================================================================================
```

**Keep this terminal open!** 🔥

---

### **📋 Step 2: Start Cloudflare Tunnel** 🌐

Open **PowerShell Terminal 2** (new window) and chant:

```powershell
cd "C:\Users\Matthew Dee\Documents\School work\LSPU\School works\3rd year\1st sem\CSST 101\Final Project\AI Model"

& "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8000
```

**The oracle shall reveal thy public URL:**
```
+--------------------------------------------------------------------------------------------+
|  Your quick Tunnel has been created! Visit it at:                                         |
|  https://random-words-xyz123.trycloudflare.com                                            |
+--------------------------------------------------------------------------------------------+
```

**Copy that sacred URL!** 📋✨

---

### **📋 Step 3: Share with Thy Website Builders** 🎨

Send thy comrades this decree:

```
🤖 AI Hardware Assistant API

Base URL: https://your-tunnel-url.trycloudflare.com

📡 API Endpoints:
  ✅ POST /chat - Converse with the AI oracle
  ✅ POST /identify - Identify hardware from sacred images
  ✅ GET /status - Check the system's vital signs
  ✅ POST /clear/{session_id} - Purge conversation history

📖 Interactive Documentation:
  https://your-tunnel-url.trycloudflare.com/docs
  (Test all endpoints in thy browser!)

🧪 Verify the Magic Works:
  https://your-tunnel-url.trycloudflare.com/status
```

---

### **⚠️ Sacred Warnings:**

| Warning | Reason | Solution |
|---------|--------|----------|
| **Keep BOTH terminals open** | Closing stops the API/tunnel | Leave them running whilst hosting |
| **Thy PC must stay awake** | Sleep mode kills the connection | Disable sleep, keep plugged in |
| **URL changes each restart** | Free tunnel generates new URLs | Share new URL each time thou restart |
| **Internet required** | For both tunnel and Gemini API | Stable connection needed |

---

### **🎯 Alternative: Use the Batch Script!**

For easier tunnel starting, simply **double-click** `start_tunnel.bat` in thy project folder instead of typing the PowerShell command! ⚡

---

### **🧪 Testing Thy Public API:**

Once thy tunnel is active, test these sacred endpoints in thy browser:

**1. Health Check:**
```
https://your-tunnel-url.trycloudflare.com/
```
Should return: `{"status":"online","message":"AI Hardware Assistant API - Ready to serve!"...}`

**2. System Status:**
```
https://your-tunnel-url.trycloudflare.com/status
```
Should show: Gemini availability, Phi-2 status, Vision model accuracy (82.50%), GPU info

**3. Interactive Docs:**
```
https://your-tunnel-url.trycloudflare.com/docs
```
Test all endpoints directly in thy browser! Try `/chat` with message `"hello"` 💬

---

### **📱 Example Vue Integration:**

Thy website builders can use this sacred code:

```javascript
// src/api/aiApi.js
import axios from 'axios';

const API_BASE_URL = 'https://your-tunnel-url.trycloudflare.com';

export const aiApi = {
  // Chat with AI
  async chat(message, sessionId = 'default') {
    const response = await axios.post(`${API_BASE_URL}/chat`, {
      message,
      session_id: sessionId,
    });
    return response.data;
  },

  // Identify hardware
  async identifyHardware(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);
    const response = await axios.post(`${API_BASE_URL}/identify`, formData);
    return response.data;
  },

  // Get status
  async getStatus() {
    const response = await axios.get(`${API_BASE_URL}/status`);
    return response.data;
  },
};
```

---

### **🎮 Quick Hosting Checklist:**

```
□ Open PowerShell #1 → python api_server.py
□ Wait for "✅ All models loaded successfully!"
□ Open PowerShell #2 → Run cloudflared tunnel command
□ Copy the https://....trycloudflare.com URL
□ Share URL with website team
□ Test /docs endpoint works
□ Keep BOTH terminals running!
□ Thy PC stays awake and connected
```

**Victory! Thy AI now dwells in the cloud realm, accessible to all!** 🌐👑⚡

---

## 🧠 Context-Aware Chat System (Smart Hardware Recognition)

**Behold!** Thy AI system now possesses **memory and awareness**! Upload an image, and the oracle shall remember what thou hast shown it! 🎯

### 🌟 **How It Works:**

1. **Upload Hardware Image** → System identifies: `"RAM 47.5%"`
2. **Ask Natural Questions** → Type: `"what's this"` or `"tell me about this"`
3. **Smart Response** → AI explains: `"This is a RAM! (Detected with 47.5% confidence) RAM provides temporary storage..."`

### 🎭 **Trigger Phrases (Automatic Detection):**

The system activates context-aware mode when thou speakest these sacred phrases:

| Phrase | Example Response |
|--------|------------------|
| `"what's this"` | Explains the last uploaded hardware |
| `"whats this"` | (Alternative spelling works!) |
| `"what is this"` | Formal version accepted |
| `"what's that"` | Points to previous detection |
| `"whats that"` | Casual variant recognized |
| `"what is that"` | Another formal option |
| `"tell me about this"` | Detailed explanation mode |
| `"explain this"` | Technical breakdown provided |

### 💡 **Usage Examples:**

**Example 1: Upload → Ask**
```
1. Upload motherboard image
   → System detects: MOTHERBOARD (48.0%)
   
2. You type: "what's this"
   
3. AI responds:
   "This is a MOTHERBOARD! (Detected with 48.0% confidence)
   
   A motherboard is the main circuit board that connects all computer 
   components together, including the CPU, RAM, GPU, and storage devices. 
   It serves as the communication backbone of the entire system."
```

**Example 2: Multiple Uploads**
```
1. Upload GPU image → Detects: GPU (89.2%)
2. Ask: "tell me about this"
   → Explains GPU functions
   
3. Upload RAM image → Detects: RAM (47.5%)
4. Ask: "what's that"
   → Explains RAM (overwrites previous GPU context)
```

**Example 3: Regular Chat Still Works**
```
1. Upload CPU image → Detects: CPU (85.3%)
2. Ask: "hello" 
   → Normal greeting response (not asking about hardware)
3. Ask: "what's this"
   → NOW explains the CPU!
```

### 🔐 **Session Management (Multi-User Support):**

Each user gets their own **session_id** to keep conversations and hardware detections separate!

**How Sessions Work:**
```
User A uploads RAM → Stored under session_id: "user-abc-123"
User B uploads GPU → Stored under session_id: "user-xyz-789"

User A asks "what's this" → Gets RAM explanation
User B asks "what's this" → Gets GPU explanation
(They don't interfere with each other!)
```

**API Usage with Sessions:**
```javascript
// Upload image with session tracking
const formData = new FormData();
formData.append('file', imageFile);
formData.append('session_id', userSessionId);  // ← Track user context!

await fetch('/identify', { method: 'POST', body: formData });

// Chat with same session
await fetch('/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "what's this",
    session_id: userSessionId  // ← Match the upload session!
  })
});
```

### 🧹 **Clear Context:**
```
POST /clear/{session_id}
```
Clears both conversation history AND hardware detection context for that session. Fresh start! 🔄

### ⚡ **Technical Flow:**

1. **`/identify` endpoint** receives image + session_id
2. Classifies hardware (e.g., "RAM 47.5%")
3. Stores result in `last_identified_hardware[session_id]`
4. **`/chat` endpoint** receives message + session_id
5. Checks if message contains trigger phrases
6. Retrieves `last_identified_hardware[session_id]`
7. Generates detailed explanation using Gemini/Phi-2
8. Returns formatted response with confidence level

### 🎯 **Pro Tips:**

✅ **Works Offline:** Uses Phi-2 fallback if Gemini unavailable  
✅ **Smart Detection:** Only activates for specific "what's this" phrases  
✅ **Fresh Explanations:** AI generates new responses each time (not pre-defined)  
✅ **Multi-User Safe:** Each session independent (3+ users simultaneously)  
✅ **Context Persists:** Upload once, ask multiple questions about it  

**Thy AI is now truly intelligent, remembering what thou hast shown it and answering with wisdom!** 🧙‍♂️✨

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

### ⚠️ Context-Aware Chat Not Working

**Symptoms:** Upload image, type "what's this", get generic response instead of hardware explanation

**Solution 1: Check Session ID Matching**

Your website **MUST** send the same `session_id` for both image upload and chat:

```javascript
// ❌ WRONG - Different session IDs
formData.append('session_id', 'user-123');  // Upload
fetch('/chat', { body: { session_id: 'default' } });  // Chat - DIFFERENT!

// ✅ CORRECT - Same session ID
const sessionId = 'user-abc-123';
formData.append('session_id', sessionId);  // Upload
fetch('/chat', { body: { session_id: sessionId } });  // Chat - SAME!
```

**Solution 2: Enable Debug Logging**

Add these lines to see what's happening:

In `api_server.py`, the debug logs will show:
```python
[DEBUG] Session ID: 4a98a6bc-48b8-4439-bb5e-f118bfdc6a2e
[DEBUG] User message: what's this
[DEBUG] Last hardware: {'component': 'ram', 'confidence': 47.5}
[DEBUG] All stored hardware: {'4a98a6bc-...': {...}}
```

Check your API terminal output when testing!

**Solution 3: Verify Form Data**

Make sure frontend sends `session_id` as **form data** (not query param):

```javascript
// ✅ CORRECT - session_id in FormData
const formData = new FormData();
formData.append('file', imageFile);
formData.append('session_id', sessionId);  // ← Must be in FormData!
```

**Solution 4: Check Trigger Phrases**

The system only activates for specific phrases:
- ✅ "what's this", "whats this", "what is this"
- ✅ "what's that", "whats that", "what is that"
- ✅ "tell me about this", "explain this"
- ❌ "hello", "hi there" (generic chat, won't explain hardware)

---

### ⚠️ Gemini API Errors (503, 429, 404)

**Symptom:** API logs show errors like:
```
503 ServiceUnavailable
429 TooManyRequests
404 NotFound
```

**Causes & Solutions:**

| Error Code | Meaning | Solution |
|------------|---------|----------|
| **503** | Gemini temporarily unavailable | Wait 30s, Phi-2 activates automatically |
| **429** | Rate limit exceeded | Reduce request frequency, use caching |
| **404** | Invalid API key/endpoint | Check `config.py` API key is valid |

**Free Tier Limits:**
- **15 requests/minute** - Wait between rapid calls
- **1,500 requests/day** - Monitor daily usage
- **1M requests/month** - Track monthly quota

**Rate Limit Prevention:**

```python
# Add delay between requests (in your website)
import time
time.sleep(4)  # Wait 4 seconds = 15 req/min max
```

**Good News:** Phi-2 fallback handles errors automatically! Users still get responses! ✅

---

### ⚠️ API Server Won't Start

**Symptom:** `python api_server.py` fails

**Solution 1: Check Dependencies**
```bash
pip install fastapi uvicorn python-multipart
```

**Solution 2: Check Port 8000**

Another program might be using port 8000:

*Windows:*
```powershell
# Find what's using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual number)
taskkill /PID <PID> /F
```

*Linux:*
```bash
# Find what's using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>
```

**Solution 3: Use Different Port**

Edit `api_server.py` bottom line:
```python
# Change from:
uvicorn.run(app, host="0.0.0.0", port=8000)

# To:
uvicorn.run(app, host="0.0.0.0", port=8080)  # Use 8080 instead
```

---

### ⚠️ Cloudflare Tunnel Not Working

**Symptom:** `cloudflared.exe` command fails or tunnel disconnects

**Solution 1: Reinstall Cloudflared**

*Windows:*
```powershell
# Download latest version
Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile "C:\Program Files (x86)\cloudflared\cloudflared.exe"
```

**Solution 2: Check API Server Running**

Tunnel requires API server at `http://localhost:8000` first:
```
1. Terminal 1: python api_server.py  ← Start this FIRST
2. Terminal 2: cloudflared tunnel     ← Then start tunnel
```

**Solution 3: Firewall Blocking**

Windows Firewall might block cloudflared:
- Allow through Windows Defender Firewall
- Or temporarily disable firewall for testing

**Solution 4: Internet Connection**

Tunnel requires stable internet. Check:
```powershell
ping cloudflare.com
```

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

## 🏛️ Chapter IX – The Sacred Tech Stack

**Behold the mystical artifacts and ancient powers that forge this legendary AI system!** ⚔️✨

### 📚 **Languages of Power**

| Language | Version | Purpose | Sacred Runes |
|----------|---------|---------|--------------|
| **Python** | 3.11 | Core AI/ML operations, model training, API backend | 🐍 |
| **JavaScript** | ES6+ | Vue.js frontend integration (thy website) | 🌐 |

---

### 🤖 **The Three Great AI Oracles**

#### **1. Google Gemini 2.5 Flash** ☁️🌟
- **Provider:** Google AI Studio
- **Role:** Primary chat oracle (cloud-based)
- **Architecture:** Transformer-based multimodal AI
- **Context Window:** 1 million tokens (vast memory!)
- **Capabilities:** Natural language understanding, hardware explanations, reasoning
- **Rate Limits:** 
  - 15 requests/minute
  - 1,500 requests/day
  - 1,000,000 requests/month
- **Cost:** FREE (educational/personal use)

#### **2. Microsoft Phi-2** 🔧💪
- **Provider:** Hugging Face Transformers
- **Role:** Local fallback oracle (offline support)
- **Architecture:** Transformer language model
- **Parameters:** 2.7 billion
- **Precision:** FP16 (half-precision for speed)
- **VRAM Usage:** 5.4GB weights + 1-3GB runtime
- **Special Power:** Works without internet!

#### **3. Google Vision Transformer (ViT)** 👁️🎯
- **Provider:** Hugging Face Transformers
- **Base Model:** `google/vit-base-patch16-224-in21k`
- **Role:** Hardware identification from images
- **Architecture:** Vision Transformer (attention-based)
- **Training Status:** Fine-tuned by thee!
- **Validation Accuracy:** 82.50%
- **Classes Detected:** 5 (CPU, GPU, Motherboard, PSU, RAM)
- **Confidence Threshold:** 25%
- **Image Resolution:** 224×224 pixels

---

### 🛠️ **Frameworks & Sacred Libraries**

#### **Deep Learning Arsenal** ⚡
```
PyTorch 2.5.1          - Main deep learning framework (tensor operations, GPU)
Transformers 4.57.1    - Hugging Face library (pre-trained models)
torchvision            - Image preprocessing & augmentation
sentencepiece          - Tokenization for Phi-2
```

#### **Web Sorcery** 🌐
```
FastAPI                - Modern REST API framework (async support)
Uvicorn                - Lightning-fast ASGI server
Pydantic               - Data validation & serialization
python-multipart       - File upload handling (multipart/form-data)
```

#### **Image Manipulation** 🖼️
```
Pillow (PIL)           - Image loading, resizing, format conversion
```

#### **Cloud Integration** ☁️
```
google-generativeai    - Official Gemini API client
```

---

### ⚡ **Optimizations & Dark Arts**

#### **GPU Acceleration** 🚀
- **CUDA Version:** 12.1
- **Automatic Device Detection:** Switches between GPU/CPU
- **Mixed Precision Training:** FP16 for 2x speed
- **Hardware Used:** NVIDIA Turing architecture
  - GTX 1660 Super (6GB VRAM)
  - RTX 2070 (8GB VRAM)

#### **Model Loading Magic** 🧙‍♂️
```python
device_map="auto"              # Smart GPU memory allocation
torch.float16                  # Half-precision inference
Preloading at startup          # Models loaded once, kept in memory
```

#### **Data Augmentation Spells** 🎨
```python
RandomHorizontalFlip()         # Mirror images randomly
RandomRotation(15°)            # Rotate up to 15 degrees
ColorJitter()                  # Adjust brightness/contrast
```

#### **Memory Management** 💾
- Gradient accumulation: Disabled (sufficient VRAM)
- Temporary file cleanup: Auto-delete after processing
- Conversation history: Last 6 exchanges only (prevents memory bloat)
- Session isolation: Independent memory per user

#### **API Performance Tricks** ⚡
```python
async def                      # Asynchronous endpoints (non-blocking)
CORS enabled                   # Cross-origin requests for Vue
Model preloading              # Load all models at startup
Debug logging                 # Troubleshooting insights
```

---

### 🌐 **Deployment & Hosting Realm**

#### **Local Server Configuration** 🏰
```
Host: 0.0.0.0                  # Listen on all network interfaces
Port: 8000                     # HTTP endpoint
Protocol: HTTP/1.1             # Standard web protocol
```

#### **Cloud Tunnel Portal** ☁️✨
```
Service: Cloudflare Tunnel (cloudflared)
Protocol: QUIC                 # Modern UDP-based transport
Public Access: HTTPS           # Automatic SSL/TLS encryption
Static IP: Not required        # Dynamic tunnel URL
Version: 2025.8.1
```

---

### 🗄️ **Data Kingdom**

#### **Dataset Arsenal** 📊
```
Total Images: 500+             # 100+ per hardware class
Training Split: 80%            # For learning patterns
Validation Split: 20%          # For testing accuracy
Formats: JPG, PNG              # Common image formats
Resolution: 224×224            # Standardized size
Augmentation: Real-time        # Applied during training
```

#### **Model Storage** 💾
```
Trained Weights: models/best_vit_model.pth
Size: ~1GB (Git LFS tracked)
Contents: State dict, accuracy, metadata
```

#### **Session Management** 🔐
```
Storage: In-memory dictionaries (Python RAM)
Conversation History: Per session_id
Hardware Context: Per session_id
Persistence: None (resets on server restart)
```

---

### 🔄 **Architecture Patterns**

#### **1. Automatic Fallback Chain** 🔗
```
User Question
    ↓
Predefined Responses? → YES → Return instant answer
    ↓ NO
Gemini API Available? → YES → Cloud response
    ↓ NO
Phi-2 Local Model → Offline response
```

#### **2. Context-Aware Flow** 🧠
```
1. User uploads image → classify_hardware()
2. Store: last_identified_hardware[session_id] = {component, confidence}
3. User types "what's this"
4. Detect trigger phrase → retrieve context
5. Generate explanation → return formatted response
```

#### **3. REST API Endpoints** 📡
```
POST   /chat              - Chat with AI (auto-routing)
POST   /identify          - Hardware identification
GET    /status            - System health check
POST   /clear/{id}        - Clear session history
GET    /docs              - Interactive API documentation (Swagger UI)
GET    /                  - Health check endpoint
```

#### **4. Multi-User Architecture** 👥
```
Session ID: UUID v4 (unique per user)
Isolation: Separate conversations & hardware contexts
Concurrency: FastAPI handles parallel requests
Thread-safe: Dictionary operations atomic
```

---

### 📦 **Complete Dependency Scroll**

```txt
# Core Deep Learning
torch==2.5.1
torchvision
transformers==4.57.1
sentencepiece

# Image Processing
pillow

# API Framework
fastapi
uvicorn[standard]
pydantic
python-multipart

# Cloud Integration
google-generativeai

# System Requirements
CUDA 12.1 (automatic with PyTorch)
NVIDIA GPU (6-8GB VRAM recommended)
Windows 11 or Linux
Python 3.11
16GB+ RAM
```

---

### 🎯 **Sacred Statistics**

| Metric | Value | Achievement |
|--------|-------|-------------|
| **Vision Accuracy** | 82.50% | ✅ Excellent |
| **Total AI Models** | 3 | 🤖 Gemini + Phi-2 + ViT |
| **API Endpoints** | 6 | 📡 RESTful design |
| **Session Support** | Unlimited | 👥 Multi-user ready |
| **Offline Support** | YES | 🔧 Phi-2 fallback |
| **Context Memory** | Per-session | 🧠 Smart responses |
| **Training Speed** | GPU 2-3x faster | ⚡ CUDA accelerated |
| **Free Tier Limits** | 1.5k/day | ☁️ Gemini API |

---

**These mystical powers combine to create thy legendary AI Hardware Assistant!** 🏆👑⚡

---

## 📜 Quick Command Reference

### 🎯 Main Commands (What You'll Use Most)

**🪟 Windows:**
```bash
# 1. Setup (one-time)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install transformers sentencepiece pillow google-generativeai fastapi uvicorn python-multipart
python check_gpu.py

# 2. Prepare dataset
python split_dataset.py --split

# 3. Train vision model
python train_vit_tiny.py

# 4. 🌟 USE THE UNIFIED SYSTEM 🌟
python model_setup.py

# 5. 🌐 HOST AS WEB API (Optional)
python api_server.py
```

**🐧 Linux:**
```bash
# 1. Setup (one-time)
python3 -m venv ai_env
source ai_env/bin/activate
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install transformers sentencepiece pillow google-generativeai fastapi uvicorn python-multipart
python3 check_gpu.py

# 2. Prepare dataset
python3 split_dataset.py --split

# 3. Train vision model
python3 train_vit_tiny.py

# 4. 🌟 USE THE UNIFIED SYSTEM 🌟
python3 model_setup.py

# 5. 🌐 HOST AS WEB API (Optional)
python3 api_server.py
```

### 🌐 API Server Commands (Web Integration)

**Start API Server:**
```bash
# Windows
python api_server.py

# Linux
python3 api_server.py
```

**Start Cloudflare Tunnel (Public Access):**
```powershell
# Windows - Full command
& "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8000

# Or use the batch script (easier!)
.\start_tunnel.bat
```

**API Endpoints Available:**
- `GET /` - Health check
- `POST /chat` - Chat with AI (Gemini/Phi-2 automatic routing)
- `POST /identify` - Identify hardware from image
- `GET /status` - System status (models loaded, GPU info, accuracy)
- `POST /clear/{session_id}` - Clear conversation & hardware context
- `GET /docs` - Interactive API documentation (Swagger UI)

**Test API Locally:**
```bash
# Health check
curl http://localhost:8000/

# System status
curl http://localhost:8000/status

# Or open in browser
start http://localhost:8000/docs  # Windows
xdg-open http://localhost:8000/docs  # Linux
```

### 📋 Complete Workflow Summary

```
Step 1: Install dependencies ✅
Step 2: Add 20+ images per hardware category to dataset/train/ ✅
Step 3: python split_dataset.py --split ✅
Step 4: python train_vit_tiny.py ✅
Step 5: python model_setup.py ✅ ← START USING YOUR AI!

Optional - Web Integration:
Step 6: python api_server.py ✅ ← Expose as REST API
Step 7: Start cloudflare tunnel ✅ ← Public URL hosting
Step 8: Share URL with website team ✅ ← Vue/React integration
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
- ✅ **REST API development** with FastAPI for web integration
- ✅ **Context-aware AI** with session management and memory
- ✅ **Cloud deployment** with Cloudflare Tunnel for public access
- ✅ **Multi-user support** with independent session tracking

### ⚔️ What Makes This Project Legendary:
1. **THREE AI Models United** - Dual chat (Gemini + Phi-2) + Vision classification
2. **Cloud + Local Hybrid** - Gemini 2.5 API (cloud) + Phi-2 (local) for flexibility
3. **Real-world Application** - Identify actual computer hardware from images
4. **Automatic Intelligence Routing** - Smart fallback system with zero manual switching
5. **GPU Optimization** - Smart VRAM management and mixed precision training
6. **Web API Integration** - FastAPI backend exposing AI via REST endpoints
7. **Context Memory System** - Remembers uploaded hardware for natural conversations
8. **Public Deployment** - Cloudflare Tunnel for sharing with teammates/presentation
9. **Session Management** - Multi-user support with independent contexts

---

### 🤖 Machine Learning vs Software Engineering Breakdown

**Understanding What's ML and What's Not:**

#### **🔬 Machine Learning Components (60% of Project):**

**Core ML Work:**
- ✅ **Neural Network Training** ([`train_vit_tiny.py`](train_vit_tiny.py ))
  - Vision Transformer fine-tuning
  - Backpropagation & gradient descent
  - Loss calculation & accuracy metrics
  - Overfitting prevention strategies
  
- ✅ **Image Classification Pipeline**
  - Forward pass through neural network
  - Softmax probability calculation
  - Confidence threshold decision-making
  - Data augmentation (flip, rotate, color jitter)

- ✅ **Transfer Learning**
  - Using pre-trained Google ViT (ImageNet weights)
  - Fine-tuning on custom hardware dataset
  - Feature extraction from images

- ✅ **AI Model Inference**
  - Gemini API (Large Language Model)
  - Phi-2 text generation (2.7B parameters)
  - Image preprocessing (normalization, resizing)

**ML Achievements:**
- 📊 **82.50% validation accuracy** on hardware classification
- 🎯 **5-class recognition** (CPU, GPU, RAM, Motherboard, PSU)
- ⚡ **Mixed precision training** (FP16 for 2x speed)
- 🧠 **Dual-model chat system** with automatic routing

---

#### **💻 Software Engineering Components (40% of Project):**

**Supporting Infrastructure:**
- ⚙️ **Web API Development** ([`api_server.py`](api_server.py ))
  - FastAPI REST endpoints
  - HTTP request/response handling
  - CORS middleware for cross-origin
  - Session management & UUID tracking

- ⚙️ **Cloud Deployment**
  - Cloudflare Tunnel networking
  - HTTPS tunneling & SSL/TLS
  - DevOps & public URL hosting

- ⚙️ **Context System**
  - Dictionary storage for hardware context
  - String pattern matching (`"what's this"`)
  - Conversation history management
  - Multi-user session isolation

- ⚙️ **File I/O & Data Handling**
  - Image loading/saving (Pillow)
  - Model checkpoint persistence
  - Temporary file cleanup
  - Dataset organization scripts

**Engineering Achievements:**
- 🌐 **Production-ready API** with 6 RESTful endpoints
- 👥 **Multi-user support** with independent sessions
- 🔄 **Automatic failover** between cloud & local models
- 📡 **Public deployment** accessible via internet

---

### 📊 Project Composition Analysis

```
🤖 Machine Learning (60%):
   ├─ Model Training (30%)
   │  └─ Vision Transformer fine-tuning
   ├─ Model Inference (20%)
   │  └─ Gemini + Phi-2 + ViT predictions
   └─ Data Preprocessing (10%)
      └─ Augmentation & normalization

💻 Software Engineering (40%):
   ├─ Web API (20%)
   │  └─ FastAPI backend development
   ├─ Frontend Integration (10%)
   │  └─ CORS, session management
   ├─ Deployment (5%)
   │  └─ Cloudflare Tunnel hosting
   └─ Context System (5%)
      └─ Memory & session tracking
```

---

### 🎯 For CSST 101 Presentation - Emphasize These:

**ML/AI Highlights (Core Contribution):**
1. 📈 **82.50% accuracy** achieved on hardware classification
2. 🤖 **Multi-model architecture** (3 AI models working together)
3. 🧠 **Transfer learning** from ImageNet to custom dataset
4. ⚡ **GPU optimization** with mixed precision training
5. 🎨 **Data augmentation** strategies for better generalization

**Bonus Engineering Skills (Shows Real-World Readiness):**
1. 🌐 **REST API development** for production deployment
2. ☁️ **Cloud hosting** with public URL access
3. 👥 **Multi-user architecture** with session isolation
4. 🔄 **Automatic failover** for reliability
5. 📱 **Web integration** ready for frontend teams

**Key Takeaway:** Thy **core contribution** is the **Machine Learning** (training the vision model, integrating multiple AI systems). The **web infrastructure** is the **delivery mechanism** that makes thy ML usable in production! 🚀

---

### 🏆 Academic Value Statement

**This project demonstrates:**
- ✅ Understanding of modern deep learning frameworks (PyTorch)
- ✅ Practical application of transfer learning techniques
- ✅ Multi-model AI system integration skills
- ✅ Real-world deployment and production readiness
- ✅ Full-stack AI development (ML + API + Deployment)

**Perfect balance for a Computer Science ML course!** The ML foundation is solid (60%), and the engineering layer shows thou canst deploy models in real applications (40%). This combination is **exactly what industry wants!** 💼👑

---

## 🆚 Chapter X – AI Model Comparison Table

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

