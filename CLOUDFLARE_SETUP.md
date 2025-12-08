# 🌐 Cloudflare Tunnel Setup Guide - Host Your AI Model Online!

This guide will help you expose your AI model API to the internet so your Vue website can access it from anywhere.

---

## 📋 What You'll Need:

- ✅ Your PC running (with the AI models)
- ✅ Internet connection
- ✅ Cloudflare Tunnel (cloudflared)

---

## 🚀 Step-by-Step Setup:

### **Step 1: Test the API Server Locally**

First, make sure your API works locally:

```powershell
python api_server.py
```

**What you should see:**
```
🚀 Starting AI Hardware Assistant API Server...
📚 Loading AI models...
⏳ Loading Gemini 2.5 Flash...
✅ Gemini AI ready!
⏳ Loading Phi-2 (2.7B params)...
✅ Phi-2 model ready!
⏳ Loading Vision Transformer...
✅ Vision model loaded! (82.50% accuracy)
✅ All models loaded successfully!
📡 API ready at http://localhost:8000
📖 Documentation at http://localhost:8000/docs
```

**Test it:**
- Open browser: `http://localhost:8000` ✅ Should show API info
- Open browser: `http://localhost:8000/docs` ✅ Should show interactive API documentation

**Press Ctrl+C to stop the server** (we'll restart it with tunnel next)

---

### **Step 2: Install Cloudflare Tunnel (cloudflared)**

#### **Windows (Easy Method - Winget):**

```powershell
winget install --id Cloudflare.cloudflared
```

#### **Windows (Manual Download):**

1. Go to: https://github.com/cloudflare/cloudflared/releases
2. Download: `cloudflared-windows-amd64.exe`
3. Rename to: `cloudflared.exe`
4. Move to: `C:\Windows\System32\` (or add to PATH)

#### **Test Installation:**

```powershell
cloudflared --version
```

You should see something like: `cloudflared version 2024.x.x`

---

### **Step 3: Start Your API + Cloudflare Tunnel**

You need **TWO terminal windows**:

#### **Terminal 1 (PowerShell) - Start API Server:**

```powershell
cd "C:\Users\Matthew Dee\Documents\School work\LSPU\School works\3rd year\1st sem\CSST 101\Final Project\AI Model"

python api_server.py
```

Keep this running! Don't close it.

---

#### **Terminal 2 (PowerShell) - Start Cloudflare Tunnel:**

```powershell
cd "C:\Users\Matthew Dee\Documents\School work\LSPU\School works\3rd year\1st sem\CSST 101\Final Project\AI Model"

cloudflared tunnel --url http://localhost:8000
```

**You'll see output like this:**

```
2024-12-08 10:30:15 INF Starting tunnel tunnelID=...
2024-12-08 10:30:16 INF +--------------------------------------------------------------------------------------------+
2024-12-08 10:30:16 INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable): |
2024-12-08 10:30:16 INF |  https://random-words-1234.trycloudflare.com                                              |
2024-12-08 10:30:16 INF +--------------------------------------------------------------------------------------------+
```

**🎉 COPY THAT URL!** That's your public API endpoint!

Example: `https://random-words-1234.trycloudflare.com`

---

### **Step 4: Test Your Public API**

Open browser and visit:
```
https://YOUR-TUNNEL-URL.trycloudflare.com
```

You should see:
```json
{
  "status": "online",
  "message": "AI Hardware Assistant API - Ready to serve!",
  "version": "1.0",
  "endpoints": {
    "chat": "POST /chat",
    "identify": "POST /identify",
    "status": "GET /status",
    "clear_history": "POST /clear/{session_id}",
    "docs": "/docs"
  }
}
```

**✅ SUCCESS!** Your API is now publicly accessible!

---

### **Step 5: Share URL with Your Friends**

Give them:
```
API Base URL: https://YOUR-TUNNEL-URL.trycloudflare.com
API Docs: https://YOUR-TUNNEL-URL.trycloudflare.com/docs
```

They can use this in the Vue website!

---

## 📡 API Endpoints for Vue Website:

### **1. Chat Endpoint**

```javascript
POST https://YOUR-TUNNEL-URL.trycloudflare.com/chat

Body:
{
  "message": "What is a GPU?",
  "session_id": "user123"
}

Response:
{
  "response": "A GPU is a Graphics Processing Unit...",
  "source": "gemini",
  "session_id": "user123"
}
```

### **2. Hardware Identification**

```javascript
POST https://YOUR-TUNNEL-URL.trycloudflare.com/identify

Body: FormData with file (multipart/form-data)

Response:
{
  "predicted_class": "GPU",
  "confidence": 89.45,
  "is_valid": true,
  "top3": [
    {"class": "GPU", "confidence": 89.45},
    {"class": "MOTHERBOARD", "confidence": 7.32},
    {"class": "CPU", "confidence": 2.11}
  ],
  "all_probabilities": {...}
}
```

### **3. System Status**

```javascript
GET https://YOUR-TUNNEL-URL.trycloudflare.com/status

Response:
{
  "gemini_available": true,
  "phi2_loaded": true,
  "vision_model_loaded": true,
  "vision_accuracy": 82.5,
  "gpu_name": "NVIDIA GeForce RTX 2070",
  "device": "cuda"
}
```

### **4. Clear Chat History**

```javascript
POST https://YOUR-TUNNEL-URL.trycloudflare.com/clear/user123

Response:
{
  "message": "History cleared",
  "session_id": "user123"
}
```

---

## ⚠️ Important Notes:

### **🔄 Tunnel URL Changes Every Restart**

The free tunnel URL changes each time you restart `cloudflared`. 

**Solution:** Update the Vue website with the new URL each time.

**Better Solution:** Use authenticated tunnel (see below).

---

### **💻 Your PC Must Stay On**

- ✅ API server must be running (`python api_server.py`)
- ✅ Cloudflare tunnel must be running (`cloudflared tunnel --url...`)
- ✅ Your PC must be connected to internet
- ✅ Don't close the terminal windows!

---

### **🔒 For Permanent Tunnel (Optional - Advanced)**

If you want a permanent URL that doesn't change:

```powershell
# 1. Login to Cloudflare
cloudflared tunnel login

# 2. Create named tunnel
cloudflared tunnel create ai-hardware-assistant

# 3. Create config file
notepad config.yml
```

**config.yml:**
```yaml
tunnel: ai-hardware-assistant
credentials-file: C:\Users\Matthew Dee\.cloudflared\UUID.json

ingress:
  - hostname: ai.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
```

```powershell
# 4. Route DNS
cloudflared tunnel route dns ai-hardware-assistant ai.yourdomain.com

# 5. Run tunnel
cloudflared tunnel run ai-hardware-assistant
```

---

## 🎮 Quick Start Commands:

### **Every time you want to host:**

**Terminal 1:**
```powershell
cd "C:\Users\Matthew Dee\Documents\School work\LSPU\School works\3rd year\1st sem\CSST 101\Final Project\AI Model"
python api_server.py
```

**Terminal 2:**
```powershell
cd "C:\Users\Matthew Dee\Documents\School work\LSPU\School works\3rd year\1st sem\CSST 101\Final Project\AI Model"
cloudflared tunnel --url http://localhost:8000
```

**Copy the tunnel URL and share with your team!** 🎉

---

## 🐛 Troubleshooting:

### **Error: "Address already in use"**

Another program is using port 8000:

```powershell
# Kill the process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F

# Or use a different port
python api_server.py --port 8001
cloudflared tunnel --url http://localhost:8001
```

### **Error: "cloudflared: command not found"**

Cloudflared not installed or not in PATH:

```powershell
# Check installation
where cloudflared

# If not found, install with winget
winget install --id Cloudflare.cloudflared

# Restart terminal after installation
```

### **API is slow**

- Check your internet upload speed
- Models take time to load first time
- GPU inference is fast, but Gemini API has network delay

---

## 📊 Performance Tips:

- ✅ Use GPU (RTX 2070 / GTX 1660 Super) for faster vision model
- ✅ Gemini handles most chat (cloud-based, no local processing)
- ✅ Phi-2 fallback is instant (local GPU inference)
- ✅ Close other GPU-heavy programs (games, video editing)

---

## 🎉 You're Done!

Your AI model is now accessible from anywhere! Your friends can build the Vue website and connect to your API.

**Next Steps for Your Friends:**
1. Get the Cloudflare tunnel URL from you
2. Use it in their Vue app API calls
3. Build the UI while you host the backend!

🚀 Happy hosting!
