# GPU Training Setup - Complete! ✅

## Your Hardware
- **GPU**: NVIDIA GeForce GTX 1660 SUPER
- **Architecture**: Turing (Compute Capability 7.5)
- **VRAM**: 6 GB
- **CUDA Version**: 12.1
- **Mixed Precision (FP16)**: ✅ Supported

## Friend's Hardware
- **GPU**: NVIDIA GeForce RTX 2070
- **Architecture**: Turing (with dedicated Tensor Cores)
- **VRAM**: 8 GB
- **Mixed Precision**: ✅ Supported (faster with Tensor Cores)

## What Was Updated

### ✅ Installed CUDA-Enabled PyTorch
- Replaced CPU-only version with CUDA 12.1 version
- Now your GPU will be used automatically

### ✅ Training Script (`train_vit_tiny.py`) Optimizations
1. **Mixed Precision (FP16)** - Uses less memory, trains faster
2. **Increased batch size** - 32 (from 16) to utilize GPU better
3. **Pin memory** - Faster data transfer to GPU
4. **Parallel data loading** - 4 workers for faster loading
5. **Non-blocking transfers** - Better GPU utilization
6. **GPU info display** - Shows your GPU name and memory

### ✅ Testing Script (`test_vit_tiny.py`) Optimizations
1. **Mixed precision inference** - Faster predictions
2. **GPU detection** - Shows which GPU is being used

## Recommended Settings

### For GTX 1660 Super (6GB VRAM):
```python
batch_size: 32-48  # Start with 32, increase if no memory errors
mixed_precision: True  # Already enabled
num_workers: 4  # Already set
```

### For RTX 2070 (8GB VRAM):
```python
batch_size: 48-64  # Can handle larger batches
mixed_precision: True  # Even faster with Tensor Cores
num_workers: 4
```

## Training Performance Expectations

### GTX 1660 Super:
- **Speed**: ~2-3x faster than CPU
- **Batch size**: 32-48 images
- **Training time**: ~5-10 minutes per epoch (depends on dataset size)

### RTX 2070:
- **Speed**: ~3-4x faster than CPU (Tensor Cores help)
- **Batch size**: 48-64 images
- **Training time**: ~3-7 minutes per epoch

## How to Use

### 1. Check GPU (anytime):
```bash
python check_gpu.py
```

### 2. Train Model:
```bash
python train_vit_tiny.py
```

You'll see:
```
Device: cuda
GPU: NVIDIA GeForce GTX 1660 SUPER
GPU Memory: 6.0 GB
Mixed Precision: Enabled (FP16)
```

### 3. Test Model:
```bash
python test_vit_tiny.py --image path/to/image.jpg
```

## Tips for Best Performance

### If you get "Out of Memory" errors:
1. Reduce batch size in `train_vit_tiny.py`:
   ```python
   'batch_size': 24,  # or 16
   ```

2. Close other GPU-intensive programs

3. Reduce image size (currently 224x224 is optimal)

### To monitor GPU usage during training:
Open another terminal and run:
```bash
nvidia-smi -l 1
```
This shows real-time GPU utilization and memory usage.

## What's Different Between Your GPUs?

### GTX 1660 Super:
- ✅ CUDA cores for parallel processing
- ✅ Mixed precision (FP16) support
- ⚠️ No dedicated Tensor Cores (but still fast!)
- 6GB VRAM

### RTX 2070:
- ✅ CUDA cores for parallel processing
- ✅ **Dedicated Tensor Cores** (2-3x faster for AI/ML)
- ✅ Mixed precision (FP16) fully optimized
- 8GB VRAM

**Both are excellent for training!** The RTX 2070 will be ~30-40% faster due to Tensor Cores, but your GTX 1660 Super is still very capable!

## Next Steps

1. ✅ **GPU is ready** - Confirmed working
2. 📸 **Add dataset** - Place images in `dataset/train/` and `dataset/val/`
3. 🚀 **Start training** - Run `python train_vit_tiny.py`
4. 🎯 **Test results** - Use `python test_vit_tiny.py`

Your GPU setup is complete and optimized for training! 🎉
