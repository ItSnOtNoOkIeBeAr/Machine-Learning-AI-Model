# 🎯 Hardware Classification AI - Quick Demo Guide

## ✅ System Status
- **Vision Model**: Trained with 81.13% accuracy
- **Classes**: CPU, GPU, RAM, Motherboard, PSU
- **Confidence Threshold**: 60% minimum
- **Device**: CUDA (RTX 2070)

## 🚀 How to Use

### 1. Start the System
```bash
python model_setup.py
```

### 2. Identify Hardware
```
identify dataset/val/cpu/4_dc406a3527654fe2bb6ad60ba284f606.jpg
identify dataset/val/gpu/17_876d9659712c4d17990012c207fe7b55.jpg
identify dataset/val/ram/27_20451dd7b37542ed83d8a9f8ae15c47b.jpg
```

### 3. Chat with AI
Just type your message:
```
What is a GPU?
How does RAM work?
Explain computer components
```

### 4. Other Commands
- `status` - Check system status
- `help` - Show all commands
- `clear` - Reset conversation
- `quit` - Exit

## 📊 What You'll See

### ✅ Valid Hardware (>60% confidence)
```
✅ IDENTIFIED: GPU
   Confidence: 85.23%

📊 All Predictions:
   gpu          ████████████████████████████████░░░░░░░░ 85.23%
   cpu          ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  8.50%
   motherboard  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  4.20%
   ram          █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  2.07%
   psu          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.00%
```

### ⚠️ Unknown Object (<60% confidence)
```
⚠️ UNKNOWN OBJECT DETECTED
   Best guess: RAM (35.20%)
   ❌ Confidence too low (< 60%)
   
   This doesn't appear to be a computer hardware component!
```

## 🎓 Model Performance

- **Training Accuracy**: 81.13%
- **Dataset**: 102 train / 53 validation images
- **Data Augmentation**: Balanced (rotation, flip, color jitter)
- **Training Time**: 20 epochs with FP16 mixed precision

## 📝 Notes

- Low confidence on some validation images is normal - the model is conservative
- For best results, use clear, well-lit images of hardware components
- The confidence threshold (60%) helps reject non-hardware images
- You can adjust `CONFIDENCE_THRESHOLD` in `model_setup.py` if needed

## 🔧 To Retrain (if needed)

1. Add more images to `dataset/train/` folders
2. Split dataset: `python split_dataset.py --split`
3. Train model: `python train_vit_tiny.py`
4. Test: `python model_setup.py`

## 🎯 Quick Test

Run the automated test:
```bash
python test_model.py
```

This will test the model on 4 sample images and show accuracy!
