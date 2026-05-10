#!/usr/bin/env python3
import os
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm


def convert_xml_to_yolo(xml_path, img_width, img_height, class_mapping):
    """
    将 VOC XML 标注转换为 YOLO 格式
    YOLO 格式: class_id x_center y_center width height (归一化到 0-1 范围)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    yolo_annotations = []
    
    for obj in root.findall('object'):
        # 获取类别名称
        class_name = obj.find('name').text
        
        # 转换为类别 ID
        if class_name not in class_mapping:
            continue
        class_id = class_mapping[class_name]
        
        # 获取边界框坐标
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        # 转换为 YOLO 格式
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        
        # 归一化
        x_center /= img_width
        y_center /= img_height
        bbox_width /= img_width
        bbox_height /= img_height
        
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")
    
    return yolo_annotations


def process_split(split_name, datasets_root, output_root, class_mapping):
    """
    处理单个数据集分割 (train/val/test)
    """
    split_dir = os.path.join(datasets_root, split_name)
    img_dir = os.path.join(split_dir, 'img')
    xml_dir = os.path.join(split_dir, 'xml')
    
    # 创建输出目录
    output_img_dir = os.path.join(output_root, 'images', split_name)
    output_label_dir = os.path.join(output_root, 'labels', split_name)
    
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 获取所有图片文件
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\n处理 {split_name} 数据集，共 {len(img_files)} 张图片...")
    
    for img_file in tqdm(img_files, desc=f"{split_name}"):
        # 复制图片
        src_img_path = os.path.join(img_dir, img_file)
        dst_img_path = os.path.join(output_img_dir, img_file)
        shutil.copy2(src_img_path, dst_img_path)
        
        # 获取图片尺寸
        import cv2
        img = cv2.imread(src_img_path)
        if img is None:
            continue
        img_height, img_width = img.shape[:2]
        
        # 处理标注
        xml_file = os.path.splitext(img_file)[0] + '.xml'
        xml_path = os.path.join(xml_dir, xml_file)
        
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(output_label_dir, label_file)
        
        if os.path.exists(xml_path):
            yolo_annotations = convert_xml_to_yolo(xml_path, img_width, img_height, class_mapping)
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations) + '\n')
        else:
            # 创建空标注文件
            with open(label_path, 'w') as f:
                pass


def create_yolo_config(output_root, class_names, config_path):
    """
    创建 YOLO 数据集配置文件
    """
    # 构建类名字符串
    names_str = '\n'.join([f'  {i}: {name}' for i, name in enumerate(class_names)])
    
    config_content = f"""path: {output_root}
train: images/train
val: images/val
test: images/test

names:
{names_str}
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"\n创建数据集配置文件: {config_path}")


def main():
    # 路径设置
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_root = os.path.join(project_root, 'datasets')
    output_root = os.path.join(datasets_root, 'DUT-Anti-UAV-YOLO')
    config_path = os.path.join(project_root, 'configs', 'dut_anti_uav.yaml')
    
    # 类别映射
    class_names = ['UAV']
    class_mapping = {'UAV': 0}
    
    print("=" * 60)
    print("DUT-Anti-UAV VOC 格式转 YOLO 格式")
    print("=" * 60)
    
    # 处理各个数据集分割
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(datasets_root, split)
        if os.path.exists(split_dir):
            process_split(split, datasets_root, output_root, class_mapping)
        else:
            print(f"\n警告: 未找到 {split} 数据集目录，跳过...")
    
    # 创建配置文件
    create_yolo_config(output_root, class_names, config_path)
    
    print("\n" + "=" * 60)
    print("转换完成！")
    print(f"数据集输出目录: {output_root}")
    print(f"配置文件: {config_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
