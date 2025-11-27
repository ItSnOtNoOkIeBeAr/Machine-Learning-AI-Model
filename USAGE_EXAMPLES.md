# 🎯 Hardware Classification AI - Usage Examples

## Starting the System
```bash
python model_setup.py
```

---

## ✅ Example 1: Valid Hardware (High Confidence)

**Command:**
```
identify dataset/val/cpu/4_dc406a3527654fe2bb6ad60ba284f606.jpg
```

**Output:**
```
🔍 Analyzing: dataset/val/cpu/4_dc406a3527654fe2bb6ad60ba284f606.jpg
⏳ Processing...

🎯 Prediction: CPU
📊 Confidence: 85.34%

📈 Top 3 Predictions:
   1. CPU: 85.34%
   2. MOTHERBOARD: 8.21%
   3. GPU: 4.15%

🤖 AI Explanation:
   A CPU (Central Processing Unit) is the primary processor that executes
   instructions and performs calculations. It acts as the brain of the
   computer system.
```

---

## ⚠️ Example 2: Unknown Object (Low Confidence)

**Command:**
```
identify C:\Users\jaype\Downloads\random_object.jpg
```

**Output:**
```
🔍 Analyzing: C:\Users\jaype\Downloads\random_object.jpg
⏳ Processing...

⚠️ UNKNOWN OBJECT DETECTED
   Best guess: CPU (39.66%)
   ❌ Confidence too low (< 60%)

   This doesn't appear to be a computer hardware component!

📊 All Predictions:
   cpu          ███████████████░░░░░░░░░░░░░░░░░░░░░░░░░ 39.66%
   gpu          █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 13.46%
   motherboard  █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 13.58%
   psu          █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 13.97%
   ram          ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 19.32%
```

---

## 💬 Example 3: Chat Mode

**Command:**
```
hello
```

**Output:**
```
🤖 Assistant: Hello! 👋 I'm your hardware identification assistant. I can help identify computer components or answer questions about PC hardware. Try 'identify <image_path>' to classify a hardware image!
```

**Command:**
```
What is a GPU?
```

**Output:**
```
🤖 Assistant: A GPU (Graphics Processing Unit) is a specialized processor designed to handle graphics rendering and parallel computations. It's essential for gaming, video editing, 3D modeling, and machine learning tasks.
```

---

## 🛠️ Utility Commands

### Status Check
```
You: status

📊 System Status:
   Chat Model: ✅ Ready (Pre-trained Phi-2)
   Vision Model: ✅ Trained
   GPU: NVIDIA GeForce RTX 2070
```

### Help
```
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
```

### Clear History
```
You: clear

🧹 Conversation history cleared!
```

### Exit
```
You: quit

👋 Goodbye!
```

---

## 📝 Tips

1. **File paths with spaces**: Quotes are automatically handled
   ```
   identify C:\Users\My Name\Desktop\cpu.jpg
   ```

2. **Confidence threshold**: Default is 60%. Objects below this are flagged as unknown

3. **Supported formats**: .jpg, .jpeg, .png, .bmp, .webp

4. **Best results**: Use clear, well-lit images focused on the hardware component

---

## 🎓 Component Descriptions

The AI provides explanations for each identified component:

- **CPU**: Primary processor, brain of the computer
- **GPU**: Graphics processor, handles rendering and parallel tasks
- **RAM**: Temporary memory for active programs
- **Motherboard**: Main circuit board connecting all components
- **PSU**: Power supply unit, distributes power to components
