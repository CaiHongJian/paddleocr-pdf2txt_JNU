# -*- coding: utf-8 -*-
"""
处理 crop_by_yolo_with_metadata.py 的输出目录：
    1. 识别每个 title 图片的文字 -> 村名
    2. 识别每个 caption 图片的文字 -> 插图名，并复制对应插图重命名（去除开头可能多余的 'O' 或 'o'）
    3. 收集所有 txt 图片的 OCR 文本（带缩进），按页码合并成一个文件，保存为 "序号_村名.txt"
    4. 每个村落的文件夹也命名为 "序号_村名"
"""
import os
import re
import shutil
import glob
import json
from pathlib import Path

from util.ocr_utils import ocr_image_to_text, ocr_image_to_json
from util.txt_extractor import extract_text_from_ocr_json
from util.txt_merger import parse_page_and_txt_num, merge_txt_segments

# ==================== 辅助函数 ====================
def sanitize_filename(name):
    """移除或替换文件名中的非法字符"""
    illegal_chars = r'[\\/*?:"<>|]'
    name = re.sub(illegal_chars, '_', name)
    name = re.sub(r'[\x00-\x1f]', '', name)
    name = name.strip()
    if not name:
        name = "未命名"
    return name

def clean_caption_text(text):
    """
    清理 caption 文本：如果第一个字符是 'O' 或 'o' 或 '0'，则去掉该字符并去除后续空白。
    例如 "O大围村村貌" -> "大围村村貌"
    """
    if not text:
        return text
    if text[0] in ('O', 'o', '0'):
        text = text[1:].lstrip()
    return text

# ==================== 主处理函数 ====================
def process_cropped_data(
    input_root,           # Images_village_cropped 目录路径
    output_root,          # 各村OCR结果 目录路径
    temp_json_dir,        # 存放 OCR 中间 JSON 的临时目录
    device="cpu",         # "cpu" 或 "gpu"
    indent_threshold=80   # 缩进判断阈值（像素）
):
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(temp_json_dir, exist_ok=True)

    # 找出所有 title 文件夹（如 Page_022_title）
    title_folders = [
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d)) and '_title' in d
    ]
    # 按页码排序（保证处理顺序与原始PDF一致）
    title_folders.sort(key=lambda x: int(re.search(r'Page_(\d+)', x).group(1)) if re.search(r'Page_(\d+)', x) else 0)

    if not title_folders:
        print(f"在 {input_root} 下未找到任何 title 文件夹")
        return

    print(f"共发现 {len(title_folders)} 个 title 文件夹")
    
    global_idx = 1  # 序号从 1 开始

    for folder in title_folders:
        folder_path = os.path.join(input_root, folder)
        print(f"\n正在处理 [{global_idx}] {folder_path}")

        # ---------- 1. 识别 title 图片，得到村名 ----------
        title_img_path = None
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            candidates = glob.glob(os.path.join(folder_path, f"*title{ext[1:]}"))
            if candidates:
                title_img_path = candidates[0]
                break
        if not title_img_path:
            print(f"  跳过 {folder}: 未找到 title 图片")
            continue

        village_name = ocr_image_to_text(
            title_img_path,
            device=device,
            temp_json_dir=os.path.join(temp_json_dir, "title_caption")
        )
        if not village_name:
            print(f"  警告: title 图片 OCR 结果为空，使用文件夹名作为村名")
            village_name = folder.replace('_title', '')
        village_name = sanitize_filename(village_name)
        print(f"  识别村名: {village_name}")

        # 带序号的村落文件夹名
        numbered_folder_name = f"{global_idx}_{village_name}"
        village_output_dir = os.path.join(output_root, numbered_folder_name)
        os.makedirs(village_output_dir, exist_ok=True)

        # ---------- 2. 处理插图（利用 img_caption_metadata.json）----------
        metadata_path = os.path.join(folder_path, "img_caption_metadata.json")
        if not os.path.exists(metadata_path):
            print(f"  警告: 未找到元数据文件 {metadata_path}，跳过插图处理")
        else:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            pairs = metadata.get("pairs", [])
            print(f"  发现 {len(pairs)} 个 img-caption 配对")

            for pair in pairs:
                img_file = pair.get("img_filename")
                caption_file = pair.get("caption_filename")
                if not img_file or not caption_file:
                    continue
                img_path = os.path.join(folder_path, img_file)
                caption_path = os.path.join(folder_path, caption_file)

                if not os.path.exists(caption_path):
                    print(f"    跳过: caption 文件不存在 {caption_path}")
                    continue

                caption_text = ocr_image_to_text(
                    caption_path,
                    device=device,
                    temp_json_dir=os.path.join(temp_json_dir, "title_caption")
                )
                if not caption_text:
                    caption_text = "无标题插图"
                # 清理开头可能多余的 'O' 或 'o'
                caption_text = clean_caption_text(caption_text)
                caption_text = sanitize_filename(caption_text)

                if os.path.exists(img_path):
                    # 处理同村同名插图
                    dest_path = os.path.join(village_output_dir, f"{caption_text}.png")
                    counter = 1
                    while os.path.exists(dest_path):
                        dest_path = os.path.join(village_output_dir, f"{caption_text}_{counter}.png")
                        counter += 1
                    shutil.copy2(img_path, dest_path)
                    print(f"    ✓ 插图已保存: {dest_path}")
                else:
                    print(f"    警告: 插图文件不存在 {img_path}")

        # ---------- 3. 收集所有 txt 图片，合并文本 ----------
        txt_images = glob.glob(os.path.join(folder_path, "*txt_*.png"))
        if not txt_images:
            print(f"  未发现 txt 图片，跳过文本合并")
        else:
            segments = []
            for txt_img_path in txt_images:
                filename = os.path.basename(txt_img_path)
                page_num, txt_num = parse_page_and_txt_num(filename)
                if page_num == 0:
                    print(f"    警告: 无法从文件名解析页码: {filename}")
                    continue

                # OCR 并保存 JSON
                json_path = ocr_image_to_json(
                    txt_img_path,
                    os.path.join(temp_json_dir, "txt"),
                    device=device
                )
                formatted_text = extract_text_from_ocr_json(json_path, indent_threshold)
                if formatted_text:
                    segments.append({
                        'page_num': page_num,
                        'txt_num': txt_num,
                        'formatted_text': formatted_text
                    })
                    print(f"    ✓ 已提取: {filename} (页码 {page_num}, txt编号 {txt_num})")
                else:
                    print(f"    ⚠ 提取文本为空: {filename}")

            if segments:
                merged_content = merge_txt_segments(segments)
                # TXT 文件名也使用序号前缀
                output_txt_name = f"{global_idx}_{village_name}.txt"
                output_txt_path = os.path.join(village_output_dir, output_txt_name)
                with open(output_txt_path, 'w', encoding='utf-8') as f:
                    f.write(merged_content)
                print(f"    ✓ 合并文本已保存: {output_txt_path} (共 {len(segments)} 个片段)")
            else:
                print(f"    ⚠ 没有有效的 txt 片段可合并")

        global_idx += 1

    print("\n所有处理完成！")

# ==================== 主入口 ====================
if __name__ == "__main__":
    # ===== 配置参数（根据实际路径修改）=====
    INPUT_ROOT = r"Test\Images_village_cropped"          # crop_by_yolo_with_metadata 输出目录
    OUTPUT_ROOT = r"Test\各村OCR结果"                     # 最终输出目录
    TEMP_JSON_DIR = r"Test\OCR_json_Results"               # 临时 JSON 目录
    DEVICE = "cpu"                                  # "cpu" 或 "gpu"
    INDENT_THRESHOLD = 50                           # 缩进阈值（像素）

    process_cropped_data(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        temp_json_dir=TEMP_JSON_DIR,
        device=DEVICE,
        indent_threshold=INDENT_THRESHOLD
    )