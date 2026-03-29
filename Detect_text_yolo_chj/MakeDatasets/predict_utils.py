"""
predict_utils.py - 智能预测工具模块
功能：对图片目录执行YOLO预测，生成标签文件。
作者：chj
日期：2026.3.28
"""

import os
import glob
from ultralytics import YOLO
from tqdm import tqdm

def predict_and_save_labels(model_path, images_dir, labels_output_dir):
    """
    对图片目录执行YOLO预测并保存标签文件
    """
    
    # 确保标签输出目录存在
    os.makedirs(labels_output_dir, exist_ok=True)
    # 支持的格式列表
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    
    # 去重并检查
    image_files = list(set(image_files)) # 防止重复
    if not image_files:
        raise ValueError(f"在目录 {images_dir} 中未找到图片文件")

    print(f" 开始预测: 共 {len(image_files)} 张图片")
    
    # 加载YOLO模型
    model = YOLO(model_path)
    
    processed_count = 0
    
    for img_path in tqdm(image_files, desc="🧠 YOLO预测中", unit='张'):
        try:
            results = model.predict(
                source=img_path,
                save=False,
                show=False,
                verbose=False
            )
            
            # 生成标签文件名
            img_name = os.path.basename(img_path)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(labels_output_dir, label_name)
            
            # 保存标签
            with open(label_path, 'w') as f:
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i])
                        xywhn = boxes.xywhn[i].tolist()
                        x_center, y_center, width, height = xywhn
                        f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            processed_count += 1
            
        except Exception as e:
            print(f" 警告: 处理图片 {img_name} 时出错: {str(e)}")
            continue
    
    return processed_count