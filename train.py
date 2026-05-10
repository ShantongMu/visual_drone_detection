#!/usr/bin/env python3
import os
import argparse
from ultralytics import RTDETR


def parse_args():
    parser = argparse.ArgumentParser(description='RT-DETR Training for Drone Detection')
    parser.add_argument('--model', type=str, default='rtdetr-l.pt',
                        help='RT-DETR model size (rtdetr-n.pt, rtdetr-s.pt, rtdetr-m.pt, rtdetr-l.pt, rtdetr-x.pt)')
    parser.add_argument('--data', type=str, default='configs/dut_anti_uav.yaml',
                        help='Path to dataset config file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--device', type=str, default='0',
                        help='Device (0 for GPU, cpu for CPU)')
    parser.add_argument('--project', type=str, default='results',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='rtdetr_drone',
                        help='Experiment name')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')

    
    return parser.parse_args()


def main():
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
    model = RTDETR(args.model)
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        exist_ok=True
    )
    
    print()
    print("=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
