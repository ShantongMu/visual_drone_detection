#!/usr/bin/env python3
import os
import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='RT-DETR Training for Drone Detection')
    parser.add_argument('--model', type=str, default='rtdetr-l.pt',
                        help='YOLO/RT-DETR model (yolov8n/s/m/l/x.pt, rtdetr-n/s/m/l/x.pt, or custom path)')
    parser.add_argument('--data', type=str, default='configs/dut_anti_uav.yaml',
                        help='Path to dataset config file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=6,
                        help='Batch size (8-24 recommended for 2x RTX 3090)')
    parser.add_argument('--imgsz', type=int, default=1280,
                        help='Image size (1280 recommended for small drone targets)')
    parser.add_argument('--device', type=str, default='0,1',
                        help='Device (0 for single GPU, 0,1 for multi-GPU, cpu for CPU)')
    parser.add_argument('--project', type=str, default='results',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='rtdetr_drone',
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
    parser.add_argument('--lr0', type=float, default=0.0002,
                        help='Initial learning rate (lowered from default for stability)')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help='Optimizer type (AdamW recommended)')
    
    return parser.parse_args()


def main():
    os.environ['NCCL_P2P_DISABLE'] = '1'
    args = parse_args()
    
    print("=" * 60)
    print("RT-DETR Drone Detection Training")
    print("=" * 60)
    print()
    print(f"Model: {args.model}")
    print(f"Data config: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print()
    
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
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
        lr0=args.lr0,
        optimizer=args.optimizer
    )
    
    print()
    print("=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
