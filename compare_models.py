#!/usr/bin/env python3
"""
YOLOv8s vs RT-DETR-L Model Comparison on Test Set
"""

import argparse
import os
import time
import json
import csv
from pathlib import Path
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8s vs RT-DETR-L Model Comparison')
    parser.add_argument('--output_dir', type=str, default='compare_results',
                        help='Directory to save comparison results')
    parser.add_argument('--data', type=str, default='configs/dut_anti_uav.yaml',
                        help='Path to dataset config file')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=1280,
                        help='Image size')
    parser.add_argument('--device', type=str, default='0',
                        help='Device (0 for GPU, cpu for CPU)')
    return parser.parse_args()


def evaluate_model(model_path, model_name, data_config, batch, imgsz, device, split='test'):
    """Evaluate a single model on test set"""
    print(f"\n{'=' * 60}")
    print(f"Evaluating {model_name}")
    print(f"{'=' * 60}")
    print(f"Model: {model_path}")
    
    start_time = time.time()
    
    model = YOLO(model_path)
    
    metrics = model.val(
        data=data_config,
        batch=batch,
        imgsz=imgsz,
        device=device,
        split=split,
        verbose=True
    )
    
    eval_time = time.time() - start_time
    
    results = {
        'model_name': model_name,
        'model_path': model_path,
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'map50': float(metrics.box.map50),
        'map50_95': float(metrics.box.map),
        'eval_time': eval_time,
        'map_per_class': metrics.box.map_per_class.tolist() if hasattr(metrics.box, 'map_per_class') else [],
        'map50_per_class': metrics.box.map50_per_class.tolist() if hasattr(metrics.box, 'map50_per_class') else [],
    }
    
    print(f"\n{model_name} Results:")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  mAP50: {results['map50']:.4f}")
    print(f"  mAP50-95: {results['map50_95']:.4f}")
    print(f"  Evaluation Time: {results['eval_time']:.2f}s")
    
    return results


def create_comparison_table(results_list, output_dir):
    """Create comparison table and save to CSV"""
    csv_path = output_dir / 'comparison_table.csv'
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Model', 'Precision', 'Recall', 'mAP50', 'mAP50-95', 
            'Eval Time (s)', 'Precision %', 'Recall %', 'mAP50 %', 'mAP50-95 %'
        ])
        
        for result in results_list:
            writer.writerow([
                result['model_name'],
                f"{result['precision']:.4f}",
                f"{result['recall']:.4f}",
                f"{result['map50']:.4f}",
                f"{result['map50_95']:.4f}",
                f"{result['eval_time']:.2f}",
                f"{result['precision']*100:.2f}%",
                f"{result['recall']*100:.2f}%",
                f"{result['map50']*100:.2f}%",
                f"{result['map50_95']*100:.2f}%",
            ])
    
    print(f"\nComparison table saved to: {csv_path}")
    return csv_path


def plot_comparison_charts(results_list, output_dir):
    """Plot comparison charts"""
    
    models = [r['model_name'] for r in results_list]
    precisions = [r['precision'] for r in results_list]
    recalls = [r['recall'] for r in results_list]
    map50s = [r['map50'] for r in results_list]
    map50_95s = [r['map50_95'] for r in results_list]
    eval_times = [r['eval_time'] for r in results_list]
    
    colors = ['#FF6B6B', '#4ECDC4']
    
    # 1. Bar chart for metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('YOLOv8s vs RT-DETR-L Model Comparison', fontsize=16, fontweight='bold')
    
    x = np.arange(len(models))
    width = 0.5
    
    # Precision
    axes[0, 0].bar(x, precisions, width, color=colors, edgecolor='black')
    axes[0, 0].set_ylabel('Precision', fontsize=12)
    axes[0, 0].set_title('Precision Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, fontsize=11)
    axes[0, 0].set_ylim([0, 1.1])
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Recall
    axes[0, 1].bar(x, recalls, width, color=colors, edgecolor='black')
    axes[0, 1].set_ylabel('Recall', fontsize=12)
    axes[0, 1].set_title('Recall Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, fontsize=11)
    axes[0, 1].set_ylim([0, 1.1])
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # mAP50
    axes[1, 0].bar(x, map50s, width, color=colors, edgecolor='black')
    axes[1, 0].set_ylabel('mAP50', fontsize=12)
    axes[1, 0].set_title('mAP50 Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models, fontsize=11)
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # mAP50-95
    axes[1, 1].bar(x, map50_95s, width, color=colors, edgecolor='black')
    axes[1, 1].set_ylabel('mAP50-95', fontsize=12)
    axes[1, 1].set_title('mAP50-95 Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models, fontsize=11)
    axes[1, 1].set_ylim([0, 1.1])
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    for i, ax in enumerate(axes.flat):
        for j, v in enumerate([precisions, recalls, map50s, map50_95s][i]):
            ax.text(j, v + 0.02, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'metrics_comparison.pdf', dpi=300, bbox_inches='tight')
    print(f"Metrics comparison chart saved to: {output_dir / 'metrics_comparison.png'}")
    
    # 2. Radar chart
    fig = plt.figure(figsize=(10, 10))
    categories = ['Precision', 'Recall', 'mAP50', 'mAP50-95']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax = fig.add_subplot(111, polar=True)
    
    for i, result in enumerate(results_list):
        values = [
            result['precision'],
            result['recall'],
            result['map50'],
            result['map50_95']
        ]
        values += values[:1]
        
        ax.plot(angles, values, color=colors[i], linewidth=3, label=result['model_name'], marker='o', markersize=8)
        ax.fill(angles, values, color=colors[i], alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title('Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_chart.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'radar_chart.pdf', dpi=300, bbox_inches='tight')
    print(f"Radar chart saved to: {output_dir / 'radar_chart.png'}")
    
    # 3. Evaluation time comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, eval_times, width, color=colors, edgecolor='black')
    ax.set_ylabel('Evaluation Time (seconds)', fontsize=12)
    ax.set_title('Evaluation Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    for j, v in enumerate(eval_times):
        ax.text(j, v + max(eval_times)*0.02, f'{v:.2f}s', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'eval_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'eval_time_comparison.pdf', dpi=300, bbox_inches='tight')
    print(f"Evaluation time chart saved to: {output_dir / 'eval_time_comparison.png'}")
    
    # 4. Combined metrics bar chart (all in one)
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(categories))
    width = 0.35
    
    yolo_values = [precisions[0], recalls[0], map50s[0], map50_95s[0]]
    rtdetr_values = [precisions[1], recalls[1], map50s[1], map50_95s[1]]
    
    bars1 = ax.bar(x - width/2, yolo_values, width, label='YOLOv8s', color='#FF6B6B', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, rtdetr_values, width, label='RT-DETR-L', color='#4ECDC4', edgecolor='black', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Complete Performance Comparison', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'complete_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'complete_comparison.pdf', dpi=300, bbox_inches='tight')
    print(f"Complete comparison chart saved to: {output_dir / 'complete_comparison.png'}")
    
    plt.close('all')


def save_json_results(results_list, output_dir):
    """Save results to JSON file"""
    json_path = output_dir / 'comparison_results.json'
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to JSON: {json_path}")
    return json_path


def print_summary(results_list):
    """Print summary comparison"""
    print(f"\n{'=' * 80}")
    print("SUMMARY: YOLOv8s vs RT-DETR-L")
    print(f"{'=' * 80}")
    
    yolo = results_list[0]
    rtdetr = results_list[1]
    
    print(f"\n{'Metric':<20} {'YOLOv8s':<15} {'RT-DETR-L':<15} {'Better':<15}")
    print(f"{'-' * 80}")
    
    metrics = [
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('mAP50', 'map50'),
        ('mAP50-95', 'map50_95'),
    ]
    
    for name, key in metrics:
        v1 = yolo[key]
        v2 = rtdetr[key]
        better = 'YOLOv8s' if v1 > v2 else 'RT-DETR-L'
        print(f"{name:<20} {v1:<15.4f} {v2:<15.4f} {better:<15}")
    
    print(f"\n{'=' * 80}")


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define models
    models = [
        {
            'name': 'YOLOv8s',
            'path': 'runs/detect/results/rtdetr_drone_safe/weights/best.pt'
        },
        {
            'name': 'RT-DETR-L',
            'path': 'runs/detect/results/rtdetr_drone/weights/best.pt'
        }
    ]
    
    all_results = []
    
    print(f"{'=' * 80}")
    print("YOLOv8s vs RT-DETR-L MODEL COMPARISON ON TEST SET")
    print(f"{'=' * 80}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Dataset config: {args.data}")
    
    # Evaluate each model
    for model in models:
        if not Path(model['path']).exists():
            print(f"\nWARNING: Model {model['path']} not found!")
            continue
        
        result = evaluate_model(
            model_path=model['path'],
            model_name=model['name'],
            data_config=args.data,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            split='test'
        )
        all_results.append(result)
    
    if len(all_results) < 2:
        print("\nERROR: Need at least 2 models to compare!")
        return
    
    # Generate outputs
    print(f"\n{'=' * 80}")
    print("GENERATING COMPARISON OUTPUTS")
    print(f"{'=' * 80}")
    
    # Save to JSON
    save_json_results(all_results, output_dir)
    
    # Create CSV table
    create_comparison_table(all_results, output_dir)
    
    # Plot charts
    plot_comparison_charts(all_results, output_dir)
    
    # Print summary
    print_summary(all_results)
    
    print(f"\n{'=' * 80}")
    print("ALL COMPARISON OUTPUTS GENERATED SUCCESSFULLY!")
    print(f"Results saved in: {output_dir.absolute()}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
