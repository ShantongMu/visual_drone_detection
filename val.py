#!/usr/bin/env python3
import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='RT-DETR Validation for Drone Detection')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pt file)')
    parser.add_argument('--data', type=str, default='configs/dut_anti_uav.yaml',
                        help='Path to dataset config file')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=1280,
                        help='Image size (1280 recommended for small drone targets)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device (0 for GPU, cpu for CPU)')
    parser.add_argument('--split', type=str, default='val',
                        help='Dataset split to validate (val or test)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("RT-DETR Drone Detection Validation")
    print("=" * 60)
    print()
    print(f"Model: {args.model}")
    print(f"Data config: {args.data}")
    print(f"Split: {args.split}")
    print()
    
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    
    metrics = model.val(
        data=args.data,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        split=args.split
    )
    
    print()
    print("=" * 60)
    print("Validation Results:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
