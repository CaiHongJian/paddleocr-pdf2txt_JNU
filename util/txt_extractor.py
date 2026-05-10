# -*- coding: utf-8 -*-
"""
文本提取模块（带缩进判断）
基于 JSON 格式的 OCR 结果，按照文本框的左侧 X 坐标判断是否为新的段落，
并将同一段落的文字直接拼接，生成格式化文本。
"""
import json

def extract_text_from_ocr_json(json_path, indent_threshold=80):
    """
    从 OCR 输出的 JSON 文件中提取文本，并根据缩进重组段落。
    参数:
        json_path: OCR 输出的 JSON 文件路径
        indent_threshold: 判断缩进的 X 坐标阈值（像素）
    返回:
        格式化后的文本字符串
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'rec_texts' not in data or 'rec_boxes' not in data:
        return ""
    
    texts = data['rec_texts']
    boxes = data['rec_boxes']  # 每个框为 [x1, y1, x2, y2]，取左上角 x1 作为缩进判断
    
    result_parts = []
    for i in range(len(texts)):
        current_text = texts[i].strip()
        if not current_text:
            continue
        
        current_left = boxes[i][0]
        is_new_paragraph = current_left >= indent_threshold
        
        if not result_parts:
            # 第一行
            if is_new_paragraph:
                result_parts.append(f"\t{current_text}")
            else:
                result_parts.append(current_text)
        else:
            if is_new_paragraph:
                result_parts.append(f"\n\t{current_text}")
            else:
                result_parts.append(current_text)
    
    return ''.join(result_parts)