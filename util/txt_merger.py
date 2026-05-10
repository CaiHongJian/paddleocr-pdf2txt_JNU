# -*- coding: utf-8 -*-
"""
文本合并模块：将同一个村子的多个 txt 片段按页码顺序合并成一个完整文本。
新增智能换行判断：若当前片段的第一行有缩进（以 '\t' 开头），则前面加换行；否则直接拼接。
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
    返回: 合并后的完整文本字符串（根据片段首行是否有缩进决定是否换行）
    """
    if not segments:
        return ""
    # 按页码升序，同一页码内按 txt_num 升序
    sorted_segments = sorted(segments, key=lambda x: (x['page_num'], x['txt_num']))
    
    result_parts = []
    for idx, seg in enumerate(sorted_segments):
        text = seg['formatted_text']
        if not text:
            continue
        
        if idx == 0:
            # 第一个片段直接添加
            result_parts.append(text)
        else:
            # 检查当前文本的第一个非空白字符是否为制表符（表示新段落）
            # 注意：首字符可能是 '\t'，也可能有换行，我们只需检查是否以 '\t' 开头
            if text.startswith('\t'):
                # 新段落，需要换行
                result_parts.append('\n' + text)
            else:
                # 连续文本，直接拼接（不加换行）
                result_parts.append(text)
    
    return ''.join(result_parts)