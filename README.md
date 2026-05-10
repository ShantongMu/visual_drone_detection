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
├── val.py                 # 验证脚本
├── inference.py           # 推理脚本
└── requirements.txt       # 依赖包列表
```

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据集准备

#### 2.1 下载数据集

访问 [DUT-Anti-UAV GitHub 仓库](https://github.com/wangdongdut/DUT-Anti-UAV) 获取数据集下载链接，或运行下载说明脚本：

```bash
python scripts/download_dut_anti_uav.py
```

将下载的数据集解压到 `datasets/DUT-Anti-UAV/` 目录，确保目录结构如下：

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

```bash
python train.py --model rtdetr-l.pt --epochs 100 --batch 8 --imgsz 640
```

常用参数：
- `--model`: RT-DETR 模型大小 (rtdetr-n.pt, rtdetr-s.pt, rtdetr-m.pt, rtdetr-l.pt, rtdetr-x.pt)
- `--data`: 数据集配置文件路径
- `--epochs`: 训练轮数
- `--batch`: 批次大小
- `--imgsz`: 输入图像尺寸
- `--device`: 训练设备 (0 表示 GPU，cpu 表示 CPU)
- `--project`: 项目目录
- `--name`: 实验名称

### 验证

```bash
python val.py --model results/rtdetr_drone/weights/best.pt --split val
```

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
