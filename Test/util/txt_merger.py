# -*- coding: utf-8 -*-
"""
文本合并模块：将同一个村子的多个 txt 片段按页码顺序合并成一个完整文本。
"""
import re

def parse_page_and_txt_num(filename):
    """
    从文件名中提取页码和 txt 编号。
    例如 "Page_022_txt_1.png" -> (22, 1)
    """
    pattern = r'Page_(\d+)_txt_(\d+)'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0

def merge_txt_segments(segments):
    """
    合并所有 txt 片段。
    segments: list of dict，每个元素包含 'page_num', 'txt_num', 'formatted_text'
    返回: 合并后的完整文本字符串（片段之间用单个换行连接，避免多余空行）
    """
    if not segments:
        return ""
    # 按页码升序，同一页码内按 txt_num 升序
    sorted_segments = sorted(segments, key=lambda x: (x['page_num'], x['txt_num']))
    # 用单个换行连接（每个片段内部已含段落换行，此处只需分隔不同片段）
    merged_text = "\n".join(seg['formatted_text'] for seg in sorted_segments if seg['formatted_text'])
    return merged_text