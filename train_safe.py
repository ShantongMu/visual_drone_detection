#!/usr/bin/env python3
import os
import argparse
from ultralytics import YOLO
from memory_safe_trainer import create_safe_trainer, MemorySafeTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Memory-Safe RT-DETR Training for Drone Detection')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                        help='YOLO/RT-DETR model (yolov8n/s/m/l/x.pt, rtdetr-n/s/m/l/x.pt, or custom path)')
    parser.add_argument('--data', type=str, default='configs/dut_anti_uav.yaml',
                        help='Path to dataset config file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Initial batch size (will auto-adjust if OOM occurs)')
    parser.add_argument('--min-batch', type=int, default=1,
                        help='Minimum batch size for auto-adjustment')
    parser.add_argument('--imgsz', type=int, default=1280,
                        help='Image size (1280 recommended for small drone targets)')
    parser.add_argument('--device', type=str, default='0,1',
                        help='Device (0 for single GPU, 0,1 for multi-GPU, cpu for CPU)')
    parser.add_argument('--project', type=str, default='results',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='rtdetr_drone_safe',
                        help='Experiment name')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loader workers (4-8 per GPU)')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Use Automatic Mixed Precision (AMP) training')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--cos-lr', action='store_true', default=True,
                        help='Use cosine learning rate scheduler')
    parser.add_argument('--memory-fraction', type=float, default=0.80,
                        help='Target GPU memory usage fraction (0.0-1.0)')
    parser.add_argument('--enable-monitor', action='store_true', default=True,
                        help='Enable real-time memory monitoring')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing (saves memory but slower)')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable deterministic mode (slower but reproducible)')
    parser.add_argument('--lr0', type=float, default=0.0002,
                        help='Initial learning rate (lowered from default for stability)')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help='Optimizer type (AdamW recommended)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Memory-Safe RT-DETR Drone Detection Training")
    print("=" * 60)
    print()
    print(f"Model: {args.model}")
    print(f"Data config: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Initial batch size: {args.batch}")
    print(f"Minimum batch size: {args.min_batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print(f"Memory fraction: {args.memory_fraction}")
    print(f"AMP enabled: {args.amp}")
    print(f"Memory monitor: {args.enable_monitor}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Learning rate (lr0): {args.lr0}")
    print()
    
    trainer = MemorySafeTrainer(
        initial_batch_size=args.batch,
        min_batch_size=args.min_batch,
        memory_fraction=args.memory_fraction,
        enable_gradient_accumulation=True,
        max_grad_accum_steps=16,
        auto_empty_cache=True
    )
    
    if args.enable_monitor and args.device != 'cpu':
        device_ids = [int(d.strip()) for d in args.device.split(',') if d.strip().isdigit()]
        for dev_id in device_ids[:1]:
            trainer.monitor_memory(dev_id)
    
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    def train_with_batch(batch, **kwargs):
        return model.train(
            data=args.data,
            epochs=args.epochs,
            batch=batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
            resume=args.resume,
            exist_ok=True,
            workers=args.workers,
            amp=args.amp,
            patience=args.patience,
            cos_lr=args.cos_lr,
            deterministic=args.deterministic,
            lr0=args.lr0,
            optimizer=args.optimizer,
            **kwargs
        )
    
    try:
        results = trainer.safe_train(
            train_func=train_with_batch,
            batch_size=args.batch,
            device=args.device
        )
        
        print()
        print("=" * 60)
        print("Training completed!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.clear_memory()
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        trainer.clear_memory()
        raise


if __name__ == "__main__":
    main()
