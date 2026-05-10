# 虚拟环境使用指南

## 环境状态
✅ **虚拟环境 `rtdetr` 创建成功！**

### 已安装的核心组件
- Python: 3.10.20
- PyTorch: 2.11.0 (CUDA 13.0 支持)
- Ultralytics: 8.4.48
- OpenCV: 4.13.0
- CUDA可用: True ✅

### 当前 Conda 环境列表
```
base                     /home/mst/miniconda3
radar5g                  /home/mst/miniconda3/envs/radar5g
rtdetr                   /home/mst/miniconda3/envs/rtdetr  ⬅️ 新环境
```

## 如何使用

### 1. 激活虚拟环境
```bash
conda activate rtdetr
```

### 2. 运行训练
```bash
conda activate rtdetr
python train.py --model rtdetr-l.pt --epochs 100 --batch 8
```

### 3. 运行推理
```bash
conda activate rtdetr
python inference.py --model <path/to/model.pt> --source <path/to/image/or/video>
```

### 4. 验证
```bash
conda activate rtdetr
python val.py --model <path/to/model.pt>
```

## 重要提示

### 不影响现有环境
- `base` 和 `radar5g` 环境完全保持不变
- 新的 `rtdetr` 环境是完全独立的

### 在环境外运行程序
如果不想每次都手动激活环境，可以使用：
```bash
conda run -n rtdetr python train.py
```

### 检查环境状态
```bash
conda info --envs
conda list -n rtdetr
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
├── val.py                 # 验证脚本
├── inference.py           # 推理脚本
├── requirements.txt       # 依赖
├── README.md              # 项目说明
└── ENV_SETUP.md           # 本文档
```
