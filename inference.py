#!/usr/bin/env python3
import os
import argparse
import cv2
from ultralytics import YOLO
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='RT-DETR Inference for Drone Detection')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                        help='Source for inference (image path, directory, video path, or 0 for webcam)')
    parser.add_argument('--output', type=str, default='results/inference',
                        help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=1280,
                        help='Image size (1280 recommended for small drone targets)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device (0 for GPU, cpu for CPU)')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save results (default: True)')
    parser.add_argument('--no-save', action='store_false', dest='save',
                        help='Do NOT save results (overrides default)')
    parser.add_argument('--show', action='store_true',
                        help='Show results (not recommended for headless environments)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("RT-DETR Drone Detection Inference")
    print("=" * 60)
    print()
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Output: {args.output}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    
    os.makedirs(args.output, exist_ok=True)
    
    print("Running inference...")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save,
        show=args.show,
        project=args.output,
        name='output',
        exist_ok=True
    )
    
    print()
    print("=" * 60)
    print(f"Inference completed! Results saved to: {args.output}/output")
    print("=" * 60)


if __name__ == "__main__":
    main()
