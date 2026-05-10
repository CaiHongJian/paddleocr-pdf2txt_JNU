# -*- coding: utf-8 -*-
"""
OCR 工具模块
功能：
    - 初始化 PaddleOCR（支持 CPU/GPU 选择）
    - 对图片进行 OCR 并保存 JSON，然后从 JSON 中提取纯文本
    - 提供直接获取纯文本的便捷函数
"""
import os
import json
from paddleocr import PaddleOCR

_ocr_instance = None

def get_ocr_instance(device="cpu"):
    """获取全局 OCR 实例（单例）"""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device
        )
    return _ocr_instance

def ocr_image_to_json(image_path, json_save_dir, device="cpu"):
    """
    对单张图片进行 OCR，将完整结果保存为 JSON 文件。
    返回: 保存的 JSON 文件路径
    """
    os.makedirs(json_save_dir, exist_ok=True)
    ocr = get_ocr_instance(device)
    result = ocr.predict(image_path)
    
    # 生成与图片同名的 JSON 文件名
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(json_save_dir, f"{base_name}.json")
    
    # 保存 JSON
    for res in result:
        res.save_to_json(json_path)
    return json_path

def ocr_image_to_text(image_path, device="cpu", temp_json_dir=None):
    """
    对单张图片进行 OCR，返回识别出的纯文本（多行用 \n 连接）。
    内部会先保存 JSON，再读取 rec_texts 字段并拼接。
    参数:
        temp_json_dir: 临时 JSON 存放目录（若不指定，会在系统临时目录下创建）
    """
    import tempfile
    if temp_json_dir is None:
        temp_json_dir = tempfile.mkdtemp(prefix="ocr_temp_")
    json_path = ocr_image_to_json(image_path, temp_json_dir, device)
    
    # 从 JSON 中读取文本
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = data.get('rec_texts', [])
    # 合并所有文本，保留原顺序，换行分隔
    return '\n'.join(texts).strip()