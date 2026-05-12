#!/usr/bin/env python3
import os
import time
import torch
import gc
from typing import Optional, Tuple, Callable


class MemorySafeTrainer:
    def __init__(self, 
                 initial_batch_size: int = 16,
                 min_batch_size: int = 1,
                 memory_fraction: float = 0.85,
                 enable_gradient_accumulation: bool = True,
                 max_grad_accum_steps: int = 8,
                 auto_empty_cache: bool = True):
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.memory_fraction = memory_fraction
        self.enable_gradient_accumulation = enable_gradient_accumulation
        self.max_grad_accum_steps = max_grad_accum_steps
        self.auto_empty_cache = auto_empty_cache
        
        self.current_batch_size = initial_batch_size
        self.grad_accum_steps = 1
        self.last_oom_time = 0
        self.oom_cooldown = 10
        
    def get_gpu_memory_info(self, device_id: int = 0) -> Tuple[int, int]:
        try:
            torch.cuda.set_device(device_id)
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(device_id)
            return allocated_memory, total_memory
        except Exception as e:
            print(f"Warning: Failed to get GPU memory info: {e}")
            return 0, 0
    
    def get_memory_usage_percent(self, device_id: int = 0) -> float:
        allocated, total = self.get_gpu_memory_info(device_id)
        if total == 0:
            return 0.0
        return (allocated / total) * 100
    
    def clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def should_reduce_batch_size(self, device_id: int = 0) -> bool:
        memory_usage = self.get_memory_usage_percent(device_id)
        return memory_usage > (self.memory_fraction * 100)
    
    def adjust_batch_size(self, device_id: int = 0) -> Tuple[int, int]:
        if self.should_reduce_batch_size(device_id) and self.current_batch_size > self.min_batch_size:
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
            print(f"Reducing batch size to {self.current_batch_size} due to high memory usage")
            self.clear_memory()
        
        if self.enable_gradient_accumulation and self.current_batch_size < self.initial_batch_size:
            target_accum = min(
                self.max_grad_accum_steps,
                self.initial_batch_size // max(1, self.current_batch_size)
            )
            if target_accum != self.grad_accum_steps:
                self.grad_accum_steps = target_accum
                print(f"Setting gradient accumulation steps to {self.grad_accum_steps}")
        
        return self.current_batch_size, self.grad_accum_steps
    
    def safe_train(self, 
                   train_func: Callable,
                   batch_size: int,
                   device: str = '0',
                   **kwargs) -> Optional[object]:
        device_ids = [int(d.strip()) for d in device.split(',') if d.strip().isdigit()]
        main_device = device_ids[0] if device_ids else 0
        
        self.current_batch_size = batch_size
        self.grad_accum_steps = 1
        
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if self.auto_empty_cache:
                    self.clear_memory()
                
                adjusted_batch, accum_steps = self.adjust_batch_size(main_device)
                
                print(f"Starting training with batch size: {adjusted_batch}, gradient accumulation: {accum_steps}")
                
                results = train_func(
                    batch=adjusted_batch,
                    **kwargs
                )
                
                print("Training completed successfully!")
                return results
                
            except RuntimeError as e:
                error_msg = str(e)
                
                if 'out of memory' in error_msg.lower() or 'cuda' in error_msg.lower():
                    retry_count += 1
                    current_time = time.time()
                    
                    if current_time - self.last_oom_time < self.oom_cooldown:
                        print(f"OOM occurred too soon, waiting {self.oom_cooldown} seconds...")
                        time.sleep(self.oom_cooldown)
                    
                    self.last_oom_time = current_time
                    
                    print(f"OOM detected (attempt {retry_count}/{max_retries})")
                    print(f"Error: {error_msg[:200]}...")
                    
                    self.clear_memory()
                    
                    if self.current_batch_size > self.min_batch_size:
                        self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
                        print(f"Reducing batch size to {self.current_batch_size}")
                    else:
                        if self.grad_accum_steps > 1:
                            self.grad_accum_steps = max(1, self.grad_accum_steps - 1)
                            print(f"Reducing gradient accumulation to {self.grad_accum_steps}")
                    
                    if retry_count >= max_retries:
                        print(f"Max retries reached. Final batch size: {self.current_batch_size}")
                        raise
                    
                    time.sleep(2)
                else:
                    raise e
            except Exception as e:
                print(f"Training error: {e}")
                raise e
    
    def monitor_memory(self, device_id: int = 0, interval: float = 5.0):
        import threading
        
        def monitor():
            while True:
                usage = self.get_memory_usage_percent(device_id)
                allocated, total = self.get_gpu_memory_info(device_id)
                allocated_gb = allocated / (1024**3)
                total_gb = total / (1024**3)
                print(f"[Memory Monitor] GPU {device_id}: {usage:.1f}% ({allocated_gb:.2f}/{total_gb:.2f} GB)")
                time.sleep(interval)
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        return thread


def create_safe_trainer(
    batch_size: int = 16,
    device: str = '0',
    enable_monitor: bool = True
) -> MemorySafeTrainer:
    trainer = MemorySafeTrainer(
        initial_batch_size=batch_size,
        min_batch_size=1,
        memory_fraction=0.80,
        enable_gradient_accumulation=True,
        max_grad_accum_steps=16,
        auto_empty_cache=True
    )
    
    if enable_monitor and torch.cuda.is_available():
        device_ids = [int(d.strip()) for d in device.split(',') if d.strip().isdigit()]
        for dev_id in device_ids[:1]:
            trainer.monitor_memory(dev_id)
    
    return trainer
