"""
Check GPU availability and CUDA setup for PyTorch
"""

import torch
import sys

def check_gpu():
    print("=" * 70)
    print("GPU & CUDA Information Check")
    print("=" * 70)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\n🔍 CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("\n❌ CUDA is not available!")
        print("\nPossible reasons:")
        print("  1. No NVIDIA GPU in system")
        print("  2. GPU drivers not installed")
        print("  3. CUDA toolkit not installed")
        print("  4. PyTorch CPU-only version installed")
        print("\nTo fix:")
        print("  - Install NVIDIA GPU drivers from: https://www.nvidia.com/download/index.aspx")
        print("  - PyTorch should auto-detect CUDA if drivers are installed")
        return
    
    # GPU Information
    print(f"\n✅ CUDA is available!")
    print(f"\n📊 CUDA Version: {torch.version.cuda}")
    print(f"📦 PyTorch Version: {torch.__version__}")
    print(f"🎮 Number of GPUs: {torch.cuda.device_count()}")
    
    # For each GPU
    for i in range(torch.cuda.device_count()):
        print(f"\n{'='*70}")
        print(f"GPU {i} Details:")
        print(f"{'='*70}")
        
        props = torch.cuda.get_device_properties(i)
        
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Multi-Processors: {props.multi_processor_count}")
        
        # Memory usage
        if torch.cuda.is_available():
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            free = (props.total_memory / 1024**3) - allocated
            
            print(f"\n  Memory Usage:")
            print(f"    Allocated: {allocated:.2f} GB")
            print(f"    Cached: {cached:.2f} GB")
            print(f"    Free: {free:.2f} GB")
    
    # Architecture information
    print(f"\n{'='*70}")
    print("Architecture Support:")
    print(f"{'='*70}")
    
    # Check for Tensor Cores (Turing/Ampere/Ada)
    if torch.cuda.is_available():
        compute_cap = torch.cuda.get_device_properties(0).major
        
        if compute_cap >= 7:
            print("  ✅ Tensor Cores: Supported")
            if compute_cap == 7:
                print("     Architecture: Turing (RTX 20 series, GTX 16 series)")
                print("     Mixed Precision (FP16): ✅ Supported")
            elif compute_cap == 8:
                print("     Architecture: Ampere (RTX 30 series)")
                print("     Mixed Precision (FP16 & TF32): ✅ Supported")
            elif compute_cap >= 9:
                print("     Architecture: Ada Lovelace / Hopper (RTX 40 series)")
                print("     Mixed Precision (FP16, TF32, FP8): ✅ Supported")
        else:
            print("  ❌ Tensor Cores: Not supported (older architecture)")
    
    # Test GPU with a simple operation
    print(f"\n{'='*70}")
    print("GPU Test:")
    print(f"{'='*70}")
    
    try:
        # Create a tensor on GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        
        # Warm up
        for _ in range(3):
            _ = torch.matmul(x, y)
        
        # Time the operation
        torch.cuda.synchronize()
        import time
        start = time.time()
        
        for _ in range(100):
            z = torch.matmul(x, y)
        
        torch.cuda.synchronize()
        end = time.time()
        
        elapsed = (end - start) / 100 * 1000  # Convert to ms
        
        print(f"  Matrix multiplication (1000x1000): {elapsed:.3f} ms per operation")
        print(f"  ✅ GPU is working correctly!")
        
        # Test mixed precision
        print(f"\n  Testing Mixed Precision (FP16):")
        with torch.cuda.amp.autocast():
            x_fp16 = x.half()
            y_fp16 = y.half()
            
            torch.cuda.synchronize()
            start = time.time()
            
            for _ in range(100):
                z = torch.matmul(x_fp16, y_fp16)
            
            torch.cuda.synchronize()
            end = time.time()
            
            elapsed_fp16 = (end - start) / 100 * 1000
            speedup = elapsed / elapsed_fp16
            
            print(f"    FP16 Performance: {elapsed_fp16:.3f} ms per operation")
            print(f"    Speedup: {speedup:.2f}x faster than FP32")
            print(f"    ✅ Mixed precision is working!")
        
    except Exception as e:
        print(f"  ❌ GPU test failed: {e}")
    
    print(f"\n{'='*70}")
    print("Recommendations for Your Setup:")
    print(f"{'='*70}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if 'gtx 1660' in gpu_name or 'gtx 16' in gpu_name:
            print("\n  GPU: GTX 1660 Super (Turing Architecture)")
            print("  ✅ Excellent for training!")
            print("  Recommended batch size: 32-64")
            print("  Mixed precision: Enabled (up to 2x speedup)")
            
        elif 'rtx 20' in gpu_name or 'rtx 2060' in gpu_name or 'rtx 2070' in gpu_name or 'rtx 2080' in gpu_name:
            print("\n  GPU: RTX 20 Series (Turing Architecture)")
            print("  ✅ Excellent for training with Tensor Cores!")
            print("  Recommended batch size: 32-64")
            print("  Mixed precision: Enabled (2-3x speedup with Tensor Cores)")
        
        if memory_gb < 6:
            print(f"\n  ⚠️ GPU Memory: {memory_gb:.1f} GB")
            print("  Consider batch size: 16-24")
        elif memory_gb < 8:
            print(f"\n  GPU Memory: {memory_gb:.1f} GB")
            print("  Recommended batch size: 32-48")
        else:
            print(f"\n  GPU Memory: {memory_gb:.1f} GB")
            print("  Recommended batch size: 48-64+")
    
    print(f"\n{'='*70}")
    print("✅ GPU check completed!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    check_gpu()
