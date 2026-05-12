# RT-DETR 无人机目标检测

基于 RT-DETR (Real-Time Detection Transformer) 的无人机目标检测项目，使用 DUT-Anti-UAV 数据集。

## 项目结构

```
rt_detr/
├── datasets/              # 数据集目录
│   ├── DUT-Anti-UAV/     # 原始数据集
│   └── DUT-Anti-UAV-YOLO/ # 转换后的 YOLO 格式数据集
├── scripts/               # 脚本目录
│   ├── download_dut_anti_uav.py    # 数据集下载说明脚本
│   └── convert_to_yolo.py          # 数据集格式转换脚本
├── configs/               # 配置文件目录
│   └── dut_anti_uav.yaml # 数据集配置
├── weights/               # 权重文件目录
├── results/               # 结果目录
├── train.py               # 训练脚本
├── train_safe.py          # 显存安全训练脚本（推荐）
├── val.py                 # 验证脚本
├── inference.py           # 推理脚本
├── memory_safe_trainer.py # 显存安全训练器
├── requirements.txt       # 依赖包列表
├── README.md              # 项目说明
├── ENV_SETUP.md           # 虚拟环境使用指南
└── MEMORY_OPTIMIZATION_GUIDE.md  # 显存优化指南
```

## 环境配置

### 1. 虚拟环境

本项目已配置 conda 虚拟环境 `mu`，包含所有必要的依赖：

| 依赖包 | 版本 | 说明 |
|--------|------|------|
| Python | 3.10.20 | 编程语言 |
| PyTorch | 2.2.0+cu121 | 深度学习框架 (CUDA 12.1) |
| Ultralytics | 8.4.48 | YOLO/RT-DETR 实现库 |
| OpenCV | 4.9.0 | 计算机视觉库 |
| NumPy | 1.26.4 | 数值计算 |
| Matplotlib | 3.8.4 | 可视化库 |

**激活虚拟环境：**
```bash
conda activate mu
```

**查看环境配置：**
```bash
conda list | grep -E "(python|torch|ultralytics|opencv)"
```

详细配置请参见 [ENV_SETUP.md](./ENV_SETUP.md)。

### 2. 数据集准备

#### 2.1 下载数据集

访问 [DUT-Anti-UAV GitHub 仓库](https://github.com/wangdongdut/DUT-Anti-UAV) 获取数据集下载链接，或运行下载说明脚本：

```bash
python scripts/download_dut_anti_uav.py
```

将下载的数据集解压到 `datasets/` 或 `datasets/DUT-Anti-UAV/` 目录。转换脚本支持多种目录结构，自动检测：

**结构 1（推荐）：**
```
datasets/
├── train/
│   ├── img/
│   └── xml/
├── val/
│   ├── img/
│   └── xml/
└── test/
    ├── img/
    └── xml/
```

**结构 2（根据 DUT-Anti-UAV 原始结构）：**
```
datasets/DUT-Anti-UAV/
├── train/
│   ├── sequences/
│   └── annotations/
├── val/
│   ├── sequences/
│   └── annotations/
└── test/
    ├── sequences/
    └── annotations/
```

#### 2.2 转换数据集格式

将 DUT-Anti-UAV 数据集转换为 YOLO 格式：

```bash
python scripts/convert_to_yolo.py
```

转换后将在 `datasets/DUT-Anti-UAV-YOLO/` 目录生成 YOLO 格式的数据集，并在 `configs/` 目录生成 `dut_anti_uav.yaml` 配置文件。

## 使用方法

### 训练

#### 显存安全训练（推荐）
使用自动显存管理的训练脚本，避免显存溢出：
```bash
python train_safe.py --model rtdetr-l.pt --epochs 100 --batch 16 --imgsz 1280
```

#### 普通训练
```bash
python train.py --model rtdetr-l.pt --epochs 100 --batch 8 --imgsz 1280
```

#### 后台运行训练（推荐用于长时间训练）

**方法一：使用 nohup（通用方式）**
```bash
cd /home/apulis-dev/code/rt_detr
nohup python train.py --model rtdetr-l.pt --epochs 100 --batch 8 --imgsz 1280 > training.log 2>&1 &
```

**方法二：使用 conda run（确保在正确环境运行）**
```bash
cd /home/apulis-dev/code/rt_detr
nohup /opt/conda/private/envs/mu/bin/python train.py --model rtdetr-l.pt --epochs 100 --batch 8 --imgsz 1280 > training.log 2>&1 &
```

**查看训练进度：**
```bash
tail -f training.log
```

**停止训练：**
```bash
# 查找训练进程
ps aux | grep train.py
# 终止进程（替换 <PID> 为实际进程号）
kill <PID>
# 强制终止（如果正常终止不起作用）
kill -9 <PID>
```

常用参数：
- `--model`: RT-DETR 模型大小 (rtdetr-n.pt, rtdetr-s.pt, rtdetr-m.pt, rtdetr-l.pt, rtdetr-x.pt)
- `--data`: 数据集配置文件路径
- `--epochs`: 训练轮数
- `--batch`: 批次大小 (根据显存调整)
- `--imgsz`: 输入图像尺寸 (**1280 推荐用于小目标无人机**，可选 1024 或更大)
- `--device`: 训练设备 (0 表示 GPU，cpu 表示 CPU)
- `--project`: 项目目录
- `--name`: 实验名称

> **重要提示：对于无人机这类小目标，建议使用 1280 或更大的 imgsz。较小的尺寸 (如 640) 会导致下采样后特征丢失。**
>
> **显存优化：** 如果遇到显存溢出问题，请使用 `train_safe.py` 或参考 [MEMORY_OPTIMIZATION_GUIDE.md](./MEMORY_OPTIMIZATION_GUIDE.md)。

### 验证

```bash
python val.py --model results/rtdetr_drone/weights/best.pt --split val
```

### 模型对比测试

使用 `compare_models.py` 可以在测试集上对比多个模型的性能：

**基本用法：**
```bash
cd /home/apulis-dev/code/rt_detr
conda run -n mu --no-capture-output python compare_models.py --device 0,1 --batch 2
```

**参数说明：**
- `--device`: GPU 设备编号（如 `0`, `0,1` 或 `cpu`）
- `--batch`: 批次大小（根据显存调整）
- `--imgsz`: 输入图像尺寸（默认 1280）
- `--output_dir`: 结果输出目录（默认 `compare_results`）

**对比结果输出：**
运行后将在 `compare_results/` 目录生成：
- `comparison_results.json` - JSON 格式的详细结果
- `comparison_table.csv` - CSV 格式的对比表格
- `metrics_comparison.png/pdf` - 指标对比柱状图
- `radar_chart.png/pdf` - 性能雷达图
- `eval_time_comparison.png/pdf` - 评估时间对比图
- `complete_comparison.png/pdf` - 完整性能对比图
- `comparison_analysis.md` - 详细分析报告

### 推理

#### 对单张图像进行推理：

```bash
python inference.py --model results/rtdetr_drone/weights/best.pt --source path/to/image.jpg
```

#### 对视频进行推理：

```bash
python inference.py --model results/rtdetr_drone/weights/best.pt --source path/to/video.mp4
```

#### 对图像目录进行推理：

```bash
python inference.py --model results/rtdetr_drone/weights/best.pt --source path/to/images/
```

## 模型选择

| 模型 | 参数量 | mAP50 (COCO) | 速度 (FPS) |
|------|--------|--------------|------------|
| RT-DETR-n | ~4.3M | ~45.7 | ~110 |
| RT-DETR-s | ~12.0M | ~50.9 | ~75 |
| RT-DETR-m | ~30.0M | ~54.8 | ~45 |
| RT-DETR-l | ~63.0M | ~56.6 | ~28 |
| RT-DETR-x | ~113M | ~57.7 | ~18 |

## 数据集说明

DUT-Anti-UAV 数据集是一个专门用于反无人机研究的数据集，包含：
- 多种场景下的无人机视频序列
- 丰富的标注信息 (边界框)
- 训练集、验证集和测试集的划分

## 参考文献

- [RT-DETR: DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)
- [DUT-Anti-UAV: A Large-Scale Benchmark for Anti-UAV Tracking](https://github.com/wangdongdut/DUT-Anti-UAV)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

## 许可证

本项目仅供学习和研究使用。
