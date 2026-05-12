
#!/usr/bin/env python3
import torch
import gc

print("Clearing GPU memory...")

# Clean up Python garbage
gc.collect()

# Clear PyTorch cache on all available GPUs
if torch.cuda.is_available():
    print(f"Found {torch.cuda.device_count()} GPU(s)")
    
    for i in range(torch.cuda.device_count()):
        try:
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            
            # Get memory info
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  Allocated: {allocated:.2f} GiB")
            print(f"  Reserved:  {reserved:.2f} GiB")
        except Exception as e:
            print(f"Error clearing GPU {i}: {e}")
else:
    print("No CUDA GPUs available")

print("\nDone!")
