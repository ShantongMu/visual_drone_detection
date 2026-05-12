# 虚拟环境使用指南

## 环境状态
✅ **虚拟环境 `mu` 已配置！**

### 已安装的核心组件
- Python: 3.10.20
- PyTorch: 2.2.0+cu121 (CUDA 12.1 支持)
- Ultralytics: 8.4.48
- OpenCV: 4.9.0

### 当前 Conda 环境列表
```
base                     /opt/conda
GBEV_G                   /opt/conda/private/envs/GBEV_G
GBVE                     /opt/conda/private/envs/GBVE
Kisaki                   /opt/conda/private/envs/Kisaki
WHX3                     /opt/conda/private/envs/WHX3
achelous                 /opt/conda/private/envs/achelous
achelous_4090            /opt/conda/private/envs/achelous_4090
drone                    /opt/conda/private/envs/drone
emotion                  /opt/conda/private/envs/emotion
isfusion                 /opt/conda/private/envs/isfusion
isfusion-5090            /opt/conda/private/envs/isfusion-5090
maritime_yolo            /opt/conda/private/envs/maritime_yolo
mu                       /opt/conda/private/envs/mu  ⬅️ 当前环境
usv_nav                  /opt/conda/private/envs/usv_nav
yolo_4090                /opt/conda/private/envs/yolo_4090
```

## 如何使用

### 1. 激活虚拟环境
```bash
conda activate mu
```

### 2. 运行训练
```bash
conda activate mu
python train.py --model rtdetr-l.pt --epochs 100 --batch 8
```

或使用显存安全版本：
```bash
conda activate mu
python train_safe.py --model rtdetr-l.pt --epochs 100 --batch 16
```

### 3. 运行推理
```bash
conda activate mu
python inference.py --model <path/to/model.pt> --source <path/to/image/or/video>
```

### 4. 验证
```bash
conda activate mu
python val.py --model <path/to/model.pt>
```

## 重要提示

### 不影响现有环境
- `mu` 环境是完全独立的，不会影响其他环境

### 在环境外运行程序
如果不想每次都手动激活环境，可以使用：
```bash
/opt/conda/private/envs/mu/bin/python train.py
```

### 检查环境状态
```bash
conda info --envs
/opt/conda/private/envs/mu/bin/python --version
/opt/conda/private/envs/mu/bin/python -c "import torch; print('PyTorch:', torch.__version__)"
```

## 项目文件结构
```
rt_detr/
├── datasets/              # 数据集目录
├── scripts/               # 工具脚本
├── configs/               # 配置文件
├── weights/               # 权重文件
├── results/               # 训练结果
├── train.py               # 训练脚本
├── train_safe.py          # 显存安全训练脚本（推荐）
├── val.py                 # 验证脚本
├── inference.py           # 推理脚本
├── memory_safe_trainer.py # 显存安全训练器
├── requirements.txt       # 依赖
├── README.md              # 项目说明
├── ENV_SETUP.md           # 本文档
└── MEMORY_OPTIMIZATION_GUIDE.md  # 显存优化指南
```
