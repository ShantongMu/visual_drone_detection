# RT-DETR 显存优化指南

## 概述
本指南提供了防止显存溢出 (OOM) 的多种优化方案，帮助你在服务器上稳定训练 RT-DETR 模型。

## 文件说明

### 1. `memory_safe_trainer.py`
显存安全训练器核心模块，包含以下功能：
- 实时显存监控
- 自动调整 batch size
- 梯度累积支持
- 自动清理显存
- OOM 自动重试机制

### 2. `train_safe.py`
安全训练脚本，集成了显存优化功能，替代原有的 `train.py`

## 使用方法

### 基础使用
```bash
python train_safe.py --model rtdetr-s.pt --data configs/dut_anti_uav.yaml
```

### 参数说明

#### 显存相关参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch` | 16 | 初始 batch size（会自动调整） |
| `--min-batch` | 1 | 最小 batch size |
| `--memory-fraction` | 0.80 | 目标显存使用率（0.0-1.0） |
| `--enable-monitor` | True | 启用实时显存监控 |

#### 训练参数（与原 train.py 相同）
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | yolov8s.pt | 模型路径 |
| `--data` | configs/... | 数据集配置 |
| `--epochs` | 100 | 训练轮数 |
| `--imgsz` | 1280 | 图像尺寸 |
| `--device` | 0,1 | GPU 设备 |
| `--amp` | True | 自动混合精度训练 |

## 显存优化策略

### 1. 自动调整 Batch Size
训练器会根据显存使用率自动降低 batch size，避免 OOM。

### 2. 梯度累积
当 batch size 减小时，自动启用梯度累积来保持有效 batch size。

### 3. 自动混合精度 (AMP)
默认启用，可减少约 50% 的显存使用。

### 4. 实时显存监控
持续显示当前显存使用情况。

## 严重显存不足时的额外措施

### 减小图像尺寸
```bash
python train_safe.py --imgsz 640 --batch 32
```

### 使用更小的模型
```bash
python train_safe.py --model rtdetr-n.pt
```

### 单 GPU 训练
```bash
python train_safe.py --device 0
```

### 减少 workers
```bash
python train_safe.py --workers 4
```

## 常见问题

### Q: 仍然出现 OOM 怎么办？
A: 尝试以下步骤：
1. 减小 `--imgsz`（如从 1280 降到 640）
2. 使用更小的模型（如 rtdetr-n）
3. 减小初始 `--batch` 值
4. 确保没有其他程序占用 GPU

### Q: 如何查看显存使用情况？
A: 
```bash
nvidia-smi
```
或者使用 `--enable-monitor` 参数（默认启用）查看实时监控。

### Q: 训练变慢了怎么办？
A: 这是正常的，因为：
- 较小的 batch size 会降低 GPU 利用率
- 梯度累积会增加计算步数
- 可以接受的话，适当增大 `--batch` 初始值

## 推荐配置

### RTX 3090 (24GB)
```bash
python train_safe.py --batch 16 --imgsz 1280
```

### RTX 3080 (10GB)
```bash
python train_safe.py --batch 8 --imgsz 1024
```

### RTX 2080 Ti (11GB)
```bash
python train_safe.py --batch 8 --imgsz 1024
```

### RTX 3060 (12GB)
```bash
python train_safe.py --batch 8 --imgsz 800
```
