# -*- coding: utf-8 -*-
"""
OCR 工具模块
功能：
    - 初始化 PaddleOCR（支持 CPU/GPU 选择）
    - 对图片进行 OCR 并保存 JSON，然后从 JSON 中提取纯文本
    - 提供直接获取纯文本的便捷函数

线程安全：每线程独立 OCR 实例，避免多线程共享导致的状态混乱
"""
import os
import json
import tempfile
import threading
import cv2
from paddleocr import PaddleOCR

# 每线程独立的 OCR 实例存储
_thread_local = threading.local()


def get_ocr_instance(device="cpu"):
    """获取当前线程的 OCR 实例（线程安全，每线程独立）"""
    key = f"ocr_{device}"
    instance = getattr(_thread_local, key, None)
    if instance is None:
        instance = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device
        )
        setattr(_thread_local, key, instance)
    return instance


def _preprocess_image_for_ocr(image_path):
    """
    图片预处理：对过小的图片进行 padding，确保 OCR 模型最小尺寸要求。
    PaddleOCR 检测模型要求输入尺寸至少为 32x32，否则特征图尺寸不匹配会报错。

    返回: 预处理后的图片路径（若是临时文件，调用者需负责清理）
    """
    MIN_SIZE = 32

    img = cv2.imread(image_path)
    if img is None:
        return image_path

    h, w = img.shape[:2]
    if h >= MIN_SIZE and w >= MIN_SIZE:
        return image_path

    # 计算 padding 量
    pad_h = max(0, MIN_SIZE - h)
    pad_w = max(0, MIN_SIZE - w)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # 用白色边框填充
    padded = cv2.copyMakeBorder(img, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # 保存为临时文件
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"_ocr_padded_{base_name}.jpg")
    cv2.imwrite(tmp_path, padded)

    return tmp_path


def ocr_image_to_json(image_path, json_save_dir, device="cpu"):
    """
    对单张图片进行 OCR，将完整结果保存为 JSON 文件。
    返回: 保存的 JSON 文件路径
    """
    os.makedirs(json_save_dir, exist_ok=True)

    # 预处理：确保图片尺寸满足 OCR 模型要求
    processed_path = _preprocess_image_for_ocr(image_path)

    try:
        ocr = get_ocr_instance(device)
        result = ocr.predict(processed_path)

        # 生成与图片同名的 JSON 文件名
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(json_save_dir, f"{base_name}.json")

        # 保存 JSON
        for res in result:
            res.save_to_json(json_path)
        return json_path
    finally:
        # 清理临时 padding 文件
        if processed_path != image_path and os.path.exists(processed_path):
            os.remove(processed_path)


def ocr_image_to_text(image_path, device="cpu", temp_json_dir=None):
    """
    对单张图片进行 OCR，返回识别出的纯文本（多行用 \n 连接）。
    内部会先保存 JSON，再读取 rec_texts 字段并拼接。
    参数:
        temp_json_dir: 临时 JSON 存放目录（若不指定，会在系统临时目录下创建）
    """
    if temp_json_dir is None:
        temp_json_dir = tempfile.mkdtemp(prefix="ocr_temp_")
    json_path = ocr_image_to_json(image_path, temp_json_dir, device)

    # 从 JSON 中读取文本
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = data.get('rec_texts', [])
    # 合并所有文本，保留原顺序，换行分隔
    return '\n'.join(texts).strip()