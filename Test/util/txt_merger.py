# -*- coding: utf-8 -*-
"""
文本合并模块
功能：将同一个村子的多个 txt 片段（来自不同页码的 txt_1/2/3.png）按页码顺序合并成一个完整文本。
"""
import re

def parse_page_and_txt_num(filename):
    """
    从文件名中提取页码和 txt 编号。
    例如 "Page_022_txt_1.png" -> (22, 1)
    返回 (page_int, txt_num_int)
    """
    # 匹配 Page_数字_txt_数字
    pattern = r'Page_(\d+)_txt_(\d+)'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    # 若匹配失败，返回默认值
    return 0, 0

def merge_txt_segments(segments):
    """
    合并所有 txt 片段。
    segments: list of dict，每个元素包含：
        - 'page_num': int
        - 'txt_num': int
        - 'formatted_text': str   (已经过缩进处理的文本)
    返回: 合并后的完整文本字符串（段落间用两个换行分隔，保持原顺序）
    """
    if not segments:
        return ""
    
    # 按页码升序，同一页码内按 txt_num 升序排序
    sorted_segments = sorted(segments, key=lambda x: (x['page_num'], x['txt_num']))
    
    # 将所有文本片段用两个换行连接（保证段落间有明显分隔）
    merged_text = "\n\n".join(seg['formatted_text'] for seg in sorted_segments if seg['formatted_text'])
    return merged_text