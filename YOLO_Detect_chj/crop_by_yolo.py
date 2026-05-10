"""
YOLO 格式数据集图片分割脚本
功能：根据 YOLO 标注坐标，将图片中的各个区域裁剪分割并保存
"""

import os
import cv2
import glob
from pathlib import Path


def load_classes(classes_file):
    """
    加载类别名称文件
    """
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    print(f"加载类别: {classes}")
    return classes


def parse_yolo_label(txt_path, img_width, img_height):
    """
    解析 YOLO 格式标注文件，返回绝对坐标
    格式: class_id x_center y_center width height
    """
    boxes = []
    
    if not os.path.exists(txt_path):
        print(f"警告: 标注文件不存在: {txt_path}")
        return boxes
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                print(f"警告: 格式错误的行: {line}")
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # 转换为绝对像素坐标
            x_center_abs = x_center * img_width
            y_center_abs = y_center * img_height
            width_abs = width * img_width
            height_abs = height * img_height
            
            # 计算左上角和右下角坐标
            x1 = int(x_center_abs - width_abs / 2)
            y1 = int(y_center_abs - height_abs / 2)
            x2 = int(x_center_abs + width_abs / 2)
            y2 = int(y_center_abs + height_abs / 2)
            
            # 边界检查
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            
            boxes.append({
                'class_id': class_id,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })
    
    return boxes


def crop_and_save(image, box, class_name, output_dir, img_name):
    """
    裁剪指定区域并保存
    """
    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
    
    # 检查裁剪区域是否有效
    if x2 <= x1 or y2 <= y1:
        print(f"  跳过无效区域: {img_name} -> {class_name} (w:{x2-x1}, h:{y2-y1})")
        return False
    
    # 裁剪图片
    cropped = image[y1:y2, x1:x2]
    
    # 生成文件名: Page_xxx_标签名.png
    output_name = f"{img_name}_{class_name}.png"
    output_path = os.path.join(output_dir, output_name)
    
    # 保存
    success = cv2.imwrite(output_path, cropped)
    
    if success:
        print(f"  ✓ 已保存: {output_name} ({x2-x1}x{y2-y1})")
        return True
    else:
        print(f"  ✗ 保存失败: {output_name}")
        return False


def process_single_image(img_path, txt_path, classes, output_dir):
    """
    处理单张图片
    """
    img_name = Path(img_path).stem  # 不含扩展名，如 Page_022
    
    # 读取图片
    image = cv2.imread(img_path)
    if image is None:
        print(f"✗ 无法读取图片: {img_path}")
        return 0
    
    img_height, img_width = image.shape[:2]
    print(f"\n处理: {img_name} ({img_width}x{img_height})")
    
    # 解析标注
    boxes = parse_yolo_label(txt_path, img_width, img_height)
    
    if not boxes:
        print(f"  未找到标注信息")
        return 0
    
    # 裁剪并保存每个区域
    saved_count = 0
    for box in boxes:
        class_id = box['class_id']
        
        # 检查类别ID是否有效
        if class_id < 0 or class_id >= len(classes):
            print(f"  警告: 无效类别ID {class_id}，跳过")
            continue
        
        class_name = classes[class_id]
        
        # 裁剪保存
        if crop_and_save(image, box, class_name, output_dir, img_name):
            saved_count += 1
    
    return saved_count


def process_dataset(
    images_dir,
    labels_dir,
    output_dir,
    classes_file,
    split='train'
):
    """
    处理整个数据集（train 或 val）
    
    参数:
        images_dir: 图片根目录 (如 Dataset_village/images)
        labels_dir: 标注根目录 (如 Dataset_village/labels)
        output_dir: 输出目录
        classes_file: classes.txt 路径
        split: 'train' 或 'val'
    """
    # 加载类别
    classes = load_classes(classes_file)
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取该 split 下的所有图片
    image_split_dir = os.path.join(images_dir, split)
    label_split_dir = os.path.join(labels_dir, split)
    
    image_files = sorted(glob.glob(os.path.join(image_split_dir, '*.png')) +
                         glob.glob(os.path.join(image_split_dir, '*.jpg')) +
                         glob.glob(os.path.join(image_split_dir, '*.jpeg')))
    
    total_images = len(image_files)
    total_crops = 0
    
    print(f"\n{'='*60}")
    print(f"开始处理 {split} 集: 共 {total_images} 张图片")
    print(f"图片路径: {image_split_dir}")
    print(f"标注路径: {label_split_dir}")
    print(f"输出路径: {output_dir}")
    print(f"{'='*60}")
    
    for idx, img_path in enumerate(image_files, 1):
        img_name = Path(img_path).stem
        txt_path = os.path.join(label_split_dir, f"{img_name}.txt")
        
        # 处理单张图片
        crops = process_single_image(img_path, txt_path, classes, output_dir)
        total_crops += crops
        
        # 进度显示
        if idx % 10 == 0 or idx == total_images:
            print(f"进度: [{idx}/{total_images}] 已裁剪 {total_crops} 个区域")
    
    print(f"\n{'='*60}")
    print(f"{split} 集处理完成!")
    print(f"处理图片: {total_images} 张")
    print(f"裁剪区域: {total_crops} 个")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")
    
    return total_crops


def process_custom_folder(
    images_dir,
    labels_dir,
    output_dir,
    classes_file
):
    """
    处理自定义文件夹（不区分 train/val，直接处理指定文件夹内的所有图片）
    
    参数:
        images_dir: 图片文件夹路径
        labels_dir: 标注文件夹路径
        output_dir: 输出文件夹路径
        classes_file: classes.txt 路径
    """
    # 加载类别
    classes = load_classes(classes_file)
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.png')) +
                         glob.glob(os.path.join(images_dir, '*.jpg')) +
                         glob.glob(os.path.join(images_dir, '*.jpeg')) +
                         glob.glob(os.path.join(images_dir, '*.bmp')) +
                         glob.glob(os.path.join(images_dir, '*.webp')))
    
    total_images = len(image_files)
    total_crops = 0
    
    print(f"\n{'='*60}")
    print(f"自定义模式: 共 {total_images} 张图片")
    print(f"图片路径: {images_dir}")
    print(f"标注路径: {labels_dir}")
    print(f"输出路径: {output_dir}")
    print(f"{'='*60}")
    
    for idx, img_path in enumerate(image_files, 1):
        img_name = Path(img_path).stem
        txt_path = os.path.join(labels_dir, f"{img_name}.txt")
        
        crops = process_single_image(img_path, txt_path, classes, output_dir)
        total_crops += crops
        
        if idx % 10 == 0 or idx == total_images:
            print(f"进度: [{idx}/{total_images}] 已裁剪 {total_crops} 个区域")
    
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"处理图片: {total_images} 张")
    print(f"裁剪区域: {total_crops} 个")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")
    
    return total_crops


def main():
    """
    主函数：配置参数并运行
    """
    # ========================================
    # 配置参数（根据你的实际情况修改）
    # ========================================
    
    # 方式一：处理标准数据集结构（train/val）
    # 数据集根目录
    DATASET_ROOT = r"Dataset_village"
    
    # 路径配置
    IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
    LABELS_DIR = os.path.join(DATASET_ROOT, "labels")
    CLASSES_FILE = os.path.join(DATASET_ROOT, "classes.txt")
    
    # 输出目录
    OUTPUT_ROOT = r"Images_village_cropped"
    
    # ========================================
    # 运行处理
    # ========================================
    
    # 处理训练集
    print("\n" + "="*60)
    print("【第一步】处理训练集")
    print("="*60)
    train_output = os.path.join(OUTPUT_ROOT, "train")
    process_dataset(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        output_dir=train_output,
        classes_file=CLASSES_FILE,
        split='train'
    )
    
    # 处理验证集
    print("\n" + "="*60)
    print("【第二步】处理验证集")
    print("="*60)
    val_output = os.path.join(OUTPUT_ROOT, "val")
    process_dataset(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        output_dir=val_output,
        classes_file=CLASSES_FILE,
        split='val'
    )
    
    # ========================================
    # 方式二：处理自定义文件夹（取消注释使用）
    # ========================================
    """
    # 自定义图片文件夹
    CUSTOM_IMAGES = r"Dataset_village\images"
    CUSTOM_LABELS = r"Dataset_village\labels"
    CUSTOM_OUTPUT = r"Images_village_cropped\custom"
    
    print("\n" + "="*60)
    print("【自定义模式】处理指定文件夹")
    print("="*60)
    process_custom_folder(
        images_dir=CUSTOM_IMAGES,
        labels_dir=CUSTOM_LABELS,
        output_dir=CUSTOM_OUTPUT,
        classes_file=CLASSES_FILE
    )
    """


if __name__ == "__main__":
    main()