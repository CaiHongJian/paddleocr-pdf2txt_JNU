"""
main.py - 专门针对图片文件夹的数据集构建主程序 (修复版)
功能：自动处理图片文件夹，支持多种格式，生成YOLO训练数据集
作者：chj (修复版)
日期：2026.3.28
"""

import os
import sys
import shutil
import glob
from tqdm import tqdm  # 导入进度条库

# 导入工具模块
from predict_utils import predict_and_save_labels
from dataset_utils import split_dataset

# ======================================================================
# ===== 配置区域（请根据实际情况修改以下参数）=====
# ======================================================================
# 【必改】输入路径
INPUT_PATH = r"Imgs"  # 图片文件夹路径

# 【必改】最终输出的数据集目录
OUTPUT_DATASET_DIR = r"newDatasets"  # 保存数据集的文件夹路径

# 【可选】YOLO模型文件路径
MODEL_PATH = r"yolo11n.pt"  

# 【可选】数据集划分比例
TRAIN_RATIO = 0.7  # 训练集比例
VAL_RATIO = 0.3   # 验证集比例
TEST_RATIO = 0.0  # 测试集比例

# 【可选】随机种子
RANDOM_SEED = 42
# ======================================================================

def main():
    # 1. 检查输入路径
    if not os.path.exists(INPUT_PATH):
        print(f"\n❌ 错误：输入路径不存在！请检查: {INPUT_PATH}")
        sys.exit(1)

    # 2. 创建临时工作目录
    base_work_dir = os.path.join(OUTPUT_DATASET_DIR, "_temp_workspace")
    temp_images_dir = os.path.join(base_work_dir, "images")
    temp_labels_dir = os.path.join(base_work_dir, "labels")
    
    # 清理旧的临时目录
    if os.path.exists(base_work_dir):
        shutil.rmtree(base_work_dir)
    
    # 创建新的临时目录
    os.makedirs(temp_images_dir, exist_ok=True)
    os.makedirs(temp_labels_dir, exist_ok=True)

    print(f"\n【步骤1】准备图片...")
    
    # 支持的格式列表
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'webp']
    image_files = []
    
    # 遍历文件夹中的所有文件，检查后缀（不区分大小写）
    for file in os.listdir(INPUT_PATH):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(INPUT_PATH, file))
    
    if not image_files:
        print(f"❌ 错误：在目录 {INPUT_PATH} 中未找到任何图片文件！")
        print("支持格式: jpg, jpeg, png, bmp, webp")
        sys.exit(1)
    
    print(f"找到 {len(image_files)} 张图片，正在复制到临时目录...")
    
    # 复制图片
    for img_path in tqdm(image_files, desc="📋 复制图片", unit='张'):
        shutil.copy2(img_path, temp_images_dir)
    
    print(f" 图片准备完成！")

    # 3. 执行YOLO预测生成标签
    print(f"\n【步骤2】执行YOLO预测生成标签文件...")
    try:
        processed = predict_and_save_labels(
            model_path=MODEL_PATH,
            images_dir=temp_images_dir,
            labels_output_dir=temp_labels_dir
        )
        print(f" 预测完成，成功生成 {processed} 个标签文件")
    except Exception as e:
        print(f"❌ 错误: 预测过程失败 - {str(e)}")
        print(f"💡 提示：请检查模型文件路径是否正确: {MODEL_PATH}")
        sys.exit(1)

    # 4. 划分数据集
    print(f"\n【步骤3】划分数据集（训练集/验证集/测试集）...")
    try:
        stats = split_dataset(
            images_dir=temp_images_dir,
            labels_dir=temp_labels_dir,
            output_dir=OUTPUT_DATASET_DIR,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            seed=RANDOM_SEED
        )
        
        print("\n" + "=" * 70)
        print("🎉 数据集构建完成！")
        print("=" * 70)
        print(f"\n📊 结果统计:")
        print(f"  总样本数: {stats['total']}")
        print(f"  训练集: {stats['train']} 张")
        print(f"  验证集: {stats['val']} 张")
        
        # 清理临时目录
        if os.path.exists(base_work_dir):
            shutil.rmtree(base_work_dir)
            
    except Exception as e:
        print(f"❌ 错误: 数据集划分失败 - {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()