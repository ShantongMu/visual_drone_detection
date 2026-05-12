
#!/usr/bin/env python3
import os
import glob

def check_labels():
    # 数据集路径 (根据yaml配置)
    dataset_path = os.path.join(os.path.dirname(__file__), 'datasets/DUT-Anti-UAV-YOLO')
    
    # 检查训练集和验证集的标签
    splits = ['train', 'val']
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Checking {split} labels...")
        print(f"{'='*60}")
        
        labels_dir = os.path.join(dataset_path, 'labels', split)
        
        if not os.path.exists(labels_dir):
            print(f"Warning: {labels_dir} does not exist!")
            continue
        
        label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
        print(f"Found {len(label_files)} label files")
        
        errors = []
        total_boxes = 0
        
        for label_file in sorted(label_files):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                try:
                    class_id = int(parts[0])
                    total_boxes += 1
                    
                    if class_id != 0:
                        errors.append({
                            'file': os.path.basename(label_file),
                            'line': line_num,
                            'class_id': class_id,
                            'line_content': line
                        })
                except (ValueError, IndexError):
                    errors.append({
                        'file': os.path.basename(label_file),
                        'line': line_num,
                        'class_id': 'INVALID',
                        'line_content': line
                    })
        
        # 打印结果
        print(f"Total boxes checked: {total_boxes}")
        print(f"Errors found: {len(errors)}")
        
        if errors:
            print(f"\nERROR DETAILS:")
            for err in errors:
                print(f"  File: {err['file']}")
                print(f"  Line: {err['line']}")
                print(f"  Class ID: {err['class_id']}")
                print(f"  Content: {err['line_content']}")
                print(f"  ---")
        else:
            print(f"\n✓ All labels are class 0!")
    
    print(f"\n{'='*60}")
    print("Check complete!")
    print(f"{'='*60}")

if __name__ == '__main__':
    check_labels()
