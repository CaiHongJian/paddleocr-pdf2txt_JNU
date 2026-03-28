# -*- coding: utf-8 -*-
"""
模块名称：ocr_module.py
功能：提供 OCR 批量处理函数
被调用者：main.py
"""

from paddleocr import PaddleOCR
import os
from tqdm import tqdm

def run_ocr(image_folder, json_output_folder):
    """
    批量处理图片并保存 JSON 结果。
    
    参数:
        image_folder (str): 存放输入图片的文件夹路径。
        json_output_folder (str): 存放输出 JSON 的文件夹路径。
    """
    # 1. 初始化 OCR 引擎
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )

    # 2. 确保输出文件夹存在
    if not os.path.exists(json_output_folder):
        os.makedirs(json_output_folder)
        print(f"【OCR模块】已创建 JSON 输出文件夹: {json_output_folder}")

    # 3. 获取图片列表
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]

    print(f"【OCR模块】开始处理 {len(image_files)} 张图片...\n")

    # 4. 批量处理
    for filename in tqdm(image_files, desc="OCR Processing", unit="image"):
        try:
            image_path = os.path.join(image_folder, filename)
            result = ocr.predict(image_path)

            # --- 核心：同名保存逻辑 ---
            name_without_ext = os.path.splitext(filename)[0]
            json_filename = f"{name_without_ext}.json"
            json_path = os.path.join(json_output_folder, json_filename)
            
            # 保存 JSON
            for res in result:
                res.save_to_json(json_path)
                
        except Exception as e:
            tqdm.write(f"处理文件 {filename} 时出错: {str(e)}")

    print("【OCR模块】✅ OCR 批量处理完成！")