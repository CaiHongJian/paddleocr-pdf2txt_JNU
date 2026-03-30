# -*- coding: utf-8 -*-
"""
json2txt_module.py：批量 JSON 转 TXT 脚本
- 功能：提供 JSON 转 TXT 提取函数
- 修复：去除了复杂的空格判断，同一段落直接拼接
"""
import json
import os
from tqdm import tqdm

def extract_text(json_input_folder, text_output_folder, indent_threshold=100):
    """
    批量读取 JSON 并提取文字保存为 TXT。
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
            result_parts = [] # 使用列表存储片段
            
            with tqdm(total=len(texts), desc=f"处理 {base_name}", unit="行", ncols=100) as pbar:
                for i in range(len(texts)):
                    current_text = texts[i].strip()
                    current_left = boxes[i][0] # 获取文本框左上角 X 坐标
                    
                    # 跳过空内容
                    if not current_text:
                        pbar.update(1)
                        continue
                        
                    # 判断是否需要缩进（即是否为新段落）
                    is_new_paragraph = current_left >= indent_threshold
                    
                    # --- 修改点：简化逻辑，直接拼接 ---
                    if not result_parts:
                        # 如果是第一行
                        if is_new_paragraph:
                            result_parts.append(f"\t{current_text}")
                        else:
                            result_parts.append(current_text)
                    else:
                        # 如果不是第一行
                        if is_new_paragraph:
                            # 新段落：换行 + 缩进
                            result_parts.append(f"\n\t{current_text}")
                        else:
                            # 同一段落：直接拼接，不做任何额外处理
                            result_parts.append(current_text)
                    
                    pbar.update(1)
            
            # 将列表合并为字符串
            result_text = ''.join(result_parts)

            # 6. 保存 TXT
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result_text)
            
            print(f"【提取模块】✅ 已生成: {output_path}")

        except Exception as e:
            print(f"【提取模块】处理 {filename} 时出错: {e}")

    print("【提取模块】✅ 所有文件提取完成！")