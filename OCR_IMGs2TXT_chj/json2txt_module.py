# -*- coding: utf-8 -*-
"""
json2txt_module.py：批量 JSON 转 TXT 脚本
- 功能：提供 JSON 转 TXT 提取函数
- 流程：读取 ocr_out_jsons 文件夹内的所有 JSON 文件 -> 提取文字并按段落处理 -> 生成同名 TXT 文件保存到 ocr_out_texts 文件夹
- 作者：蔡鸿键 (已修复首行缩进 Bug)
"""
import json
import os
from tqdm import tqdm

def extract_text(json_input_folder, text_output_folder, indent_threshold=100):
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

            # 5. 文本拼接逻辑 (已修复首行缩进 Bug)
            result_lines = [] # 使用列表存储行，最后统一 join，效率更高且逻辑更清晰
            indent_style = "\t"
            
            # 标记是否是第一行，用于处理首行缩进
            is_first_line = True
            
            with tqdm(total=len(texts), desc=f"处理 {base_name}", unit="行", ncols=100) as pbar:
                for i in range(len(texts)):
                    current_text = texts[i].strip()
                    current_left = boxes[i][0] # 获取文本框左上角 X 坐标
                    
                    # 判断是否需要缩进
                    need_indent = current_left >= indent_threshold
                    
                    # --- 【Bug 修复点】---
                    # 原逻辑：第一行直接拼接，忽略了缩进判断。
                    # 新逻辑：统一处理所有行，包括第一行。
                    
                    if is_first_line:
                        # 处理第一行
                        if need_indent:
                            result_lines.append(f"{indent_style}{current_text}")
                        else:
                            result_lines.append(current_text)
                        is_first_line = False # 更新标志位
                    else:
                        # 处理非第一行（换行逻辑）
                        if need_indent:
                            result_lines.append(f"\n{indent_style}{current_text}")
                        else:
                            result_lines.append(f"\n{current_text}")
                    
                    pbar.update(1)
            
            # 将列表合并为字符串
            result_text = ''.join(result_lines)

            # 6. 保存 TXT
            with open(output_path, 'w', encoding=' utf-8') as f:
                f.write(result_text)
            
            print(f"【提取模块】✅ 已生成: {output_path}")

        except Exception as e:
            print(f"【提取模块】处理 {filename} 时出错: {e}")

    print("【提取模块】✅ 所有文件提取完成！")