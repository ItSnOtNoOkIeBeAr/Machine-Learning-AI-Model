# Windows Long Path Fix Instructions

## The Problem
Windows has a default 260-character path length limit, and some PyTorch files have very long paths that exceed this limit.

## Solution 1: Enable Long Paths (Recommended)

### Step 1: Run PowerShell as Administrator
1. Press Windows key
2. Type "PowerShell"
3. Right-click "Windows PowerShell"
4. Select "Run as administrator"

### Step 2: Enable Long Paths
Run this command in the admin PowerShell:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### Step 3: Enable for Git (if you use Git)
```powershell
git config --system core.longpaths true
```

### Step 4: Restart Your Computer
Long paths will only take effect after a restart.

### Step 5: Install PyTorch After Restart
```bash
pip install -r requirements.txt
```

---

## Solution 2: Use a Shorter Path (Quick Alternative)

If you can't enable long paths or restart, move your project to a shorter path:

### Example:
Instead of: `c:\Users\Matthew Dee\Documents\School work\LSPU\School works\3rd year\1st sem\CSST 101\Final Project\AI Model`

Use: `c:\AI_Model` or `c:\Projects\AI_Model`

Then run:
```bash
cd c:\AI_Model
pip install -r requirements.txt
```

---

## Solution 3: Use Conda (Alternative Package Manager)

Conda handles long paths better:

### Install Miniconda
Download from: https://docs.conda.io/en/latest/miniconda.html

### Create Environment and Install
```bash
conda create -n ai_model python=3.11
conda activate ai_model
conda install pytorch -c pytorch
pip install transformers accelerate sentencepiece protobuf safetensors
```

---

## After Installation

Once you've successfully installed the packages, run:
```bash
python model_setup.py
```

This will download the AI model (~3GB) and run a test.
