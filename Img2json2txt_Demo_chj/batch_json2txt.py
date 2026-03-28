# -*- coding: utf-8 -*-
"""
批量 JSON 转 TXT 脚本（demo）
功能：读取 ocr_out_jsons 文件夹内的所有 JSON 文件 -> 提取文字并按段落处理 -> 生成同名 TXT 文件保存到 ocr_out_texts 文件夹
作者：蔡鸿键
"""

import json
import os
from tqdm import tqdm

def process_ocr_json(json_file_path, output_dir="ocr_out_texts", indent_threshold=20):
    """
    处理单个 OCR JSON 文件，提取文字并保存为 TXT。
    
    参数:
        json_file_path (str): JSON 文件的完整路径。
        output_dir (str): TXT 文件的输出目录。
        indent_threshold (int): 判断段落缩进的坐标阈值。
    """
    try:
        # 1. 读取 JSON 文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 2. 检查数据结构是否正确
        if 'rec_texts' not in data or 'rec_boxes' not in data:
            print(f"错误：文件 {json_file_path} 中缺少 'rec_texts' 或 'rec_boxes' 字段")
            return

        texts = data['rec_texts']  # 识别出的文字列表
        boxes = data['rec_boxes']  # 对应的文字框坐标列表

        # 3. 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 4. 生成输出文件名（与输入 JSON 同名）
        base_name = os.path.splitext(os.path.basename(json_file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")

        # 5. 核心处理逻辑：根据坐标拼接文本
        result_text = ""
        total_lines = len(texts)
        
        # 使用制表符作为段落缩进
        indent_style = "\t"
        
        # 创建进度条
        with tqdm(total=total_lines, desc=f"处理 {base_name}", unit="行", colour="#ffffff", ncols=100) as pbar:
            for i in range(total_lines):
                current_text = texts[i].strip()
                current_left = boxes[i][0] # 获取文字框的左上角 X 坐标

                # --- 段落判断逻辑 ---
                # 如果 X 坐标大于阈值，认为是新段落（前面加换行和缩进）
                # 如果 X 坐标小于阈值，认为是上一行的延续（直接拼接）
                if current_left >= indent_threshold:
                    result_text += f"\n{indent_style}{current_text}"
                else:
                    result_text += current_text

                # 更新进度条
                pbar.update(1)

        # 去除结果开头可能存在的多余换行或空白
        result_text = result_text.lstrip('\n').lstrip()

        # 6. 保存为 TXT 文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result_text)

        print(f"✅ 文件处理完成: {output_path}")

    except Exception as e:
        print(f"处理文件 {json_file_path} 时发生错误: {e}")

def main():
    # --- 配置区 ---
    # 输入文件夹：存放 OCR 生成的 JSON 文件
    input_folder = "ocr_out_jsons"
    
    # 支持的文件扩展名
    json_extensions = ('.json',)

    # 1. 获取输入文件夹内所有的 JSON 文件
    if not os.path.exists(input_folder):
        print(f"错误：找不到输入文件夹 {input_folder}，请先运行 OCR 脚本生成 JSON 文件。")
        return

    json_files = [f for f in os.listdir(input_folder) if f.lower().endswith(json_extensions)]
    
    if len(json_files) == 0:
        print(f"警告：在文件夹 {input_folder} 中未找到任何 JSON 文件。")
        return

    print(f"发现 {len(json_files)} 个 JSON 文件，开始批量转换...\n")

    # 2. 批量处理每一个 JSON 文件
    for filename in json_files:
        # 拼接完整的文件路径
        full_path = os.path.join(input_folder, filename)
        # 调用处理函数
        process_ocr_json(full_path)

if __name__ == "__main__":
    main()