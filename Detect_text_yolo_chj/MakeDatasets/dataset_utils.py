"""
dataset_utils.py - 数据集划分工具模块 (修复版)
功能：将图片和标签划分为训练集、验证集、测试集。支持多种图片格式。
作者：chj (修复版)
日期：2026.3.28
"""

import os
import shutil
import random
from tqdm import tqdm

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.3, test_ratio=0.0, seed=42):
    """
    将数据集划分为训练集、验证集和测试集
    """
    
    # 1. 验证比例
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"划分比例总和必须为1.0，当前总和为: {total_ratio:.4f}")
    
    random.seed(seed)
    

    # 获取所有支持的图片文件
    supported_img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []
    
    for file in os.listdir(images_dir):
        if any(file.lower().endswith(ext) for ext in supported_img_exts):
            image_files.append(file)
    
    if not image_files:
        # 这里给出更具体的报错信息，方便调试
        files_in_dir = os.listdir(images_dir)
        raise ValueError(f"在目录 {images_dir} 中未找到支持的图片文件！"
                        f"目录内容: {files_in_dir} (请检查文件格式是否被支持)")

    # 2. 检查图片-标签配对
    valid_pairs = []
    for img_file in image_files:
        # 获取文件名（不含后缀）
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + ".txt"
        label_path = os.path.join(labels_dir, label_file)
        
        if os.path.exists(label_path):
            valid_pairs.append((img_file, label_file))
        else:
            print(f"⚠️ 跳过: 找不到标签文件 {label_file} (缺少对应标签)")

    if not valid_pairs:
        raise ValueError("未找到有效的图片-标签配对文件！")
    
    # 3. 随机打乱并计算分割点
    random.shuffle(valid_pairs)
    total = len(valid_pairs)
    
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_pairs = valid_pairs[:train_end]
    val_pairs = valid_pairs[train_end:val_end]
    test_pairs = valid_pairs[val_end:]
    
    # 4. 创建输出目录结构
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
    
    # 定义复制函数
    def copy_files(pairs, split_name):
        img_out_dir = os.path.join(output_dir, 'images', split_name)
        lbl_out_dir = os.path.join(output_dir, 'labels', split_name)
        
        for img_file, lbl_file in tqdm(pairs, desc=f"📦 复制到 {split_name}", unit='文件'):
            shutil.copy2(os.path.join(images_dir, img_file), img_out_dir)
            shutil.copy2(os.path.join(labels_dir, lbl_file), lbl_out_dir)
    
    # 执行复制
    if train_pairs:
        copy_files(train_pairs, 'train')
    if val_pairs:
        copy_files(val_pairs, 'val')
    if test_pairs:
        copy_files(test_pairs, 'test')
    
    # 5. 返回统计
    return {
        'total': total,
        'train': len(train_pairs),
        'val': len(val_pairs),
        'test': len(test_pairs)
    }