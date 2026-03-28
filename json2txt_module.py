# -*- coding: utf-8 -*-
"""
模块名称：json2txt_module.py
功能：提供 JSON 转 TXT 提取函数
被调用者：main.py
"""

import json
import os
from tqdm import tqdm

def extract_text(json_input_folder, text_output_folder, indent_threshold=20):
    """
    批量读取 JSON 并提取文字保存为 TXT。
    
    参数:
        json_input_folder (str): 存放输入 JSON 的文件夹路径。
        text_output_folder (str): 存放输出 TXT 的文件夹路径。
        indent_threshold (int): 判断段落缩进的坐标阈值。
    """
    # 1. 检查输入文件夹
    if not os.path.exists(json_input_folder):
        print(f"【提取模块】错误：找不到输入文件夹 {json_input_folder}")
        return

    # 2. 确保输出文件夹存在
    if not os.path.exists(text_output_folder):
        os.makedirs(text_output_folder)
        print(f"【提取模块】已创建 TXT 输出文件夹: {text_output_folder}")

    # 3. 获取 JSON 文件列表
    json_files = [f for f in os.listdir(json_input_folder) if f.lower().endswith('.json')]
    
    if len(json_files) == 0:
        print(f"【提取模块】警告：在文件夹 {json_input_folder} 中未找到 JSON 文件。")
        return

    print(f"【提取模块】发现 {len(json_files)} 个 JSON 文件，开始批量转换...\n")

    # 4. 批量处理每个 JSON 文件
    for filename in json_files:
        json_file_path = os.path.join(json_input_folder, filename)
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'rec_texts' not in data or 'rec_boxes' not in data:
                print(f"跳过文件 {filename}：数据结构错误")
                continue

            texts = data['rec_texts']
            boxes = data['rec_boxes']

            # --- 核心：同名保存逻辑 ---
            base_name = os.path.splitext(filename)[0] # 去掉 .json 后缀
            output_path = os.path.join(text_output_folder, f"{base_name}.txt")

            # 5. 文本拼接逻辑
            result_text = ""
            indent_style = "\t"
            
            with tqdm(total=len(texts), desc=f"处理 {base_name}", unit="行", ncols=100) as pbar:
                for i in range(len(texts)):
                    current_text = texts[i].strip()
                    current_left = boxes[i][0]

                    if current_left >= indent_threshold:
                        result_text += f"\n{indent_style}{current_text}"
                    else:
                        result_text += current_text

                    pbar.update(1)

            result_text = result_text.lstrip('\n').lstrip()

            # 6. 保存 TXT
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result_text)

            print(f"【提取模块】✅ 已生成: {output_path}")

        except Exception as e:
            print(f"【提取模块】处理 {filename} 时出错: {e}")

    print("【提取模块】✅ 所有文件提取完成！")