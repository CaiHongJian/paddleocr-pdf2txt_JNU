# -*- coding: utf-8 -*-
"""
处理 crop_by_yolo_with_metadata.py 的输出目录：
    1. 识别每个 title 图片的文字 -> 村名
    2. 识别每个 caption 图片的文字 -> 插图名，并复制对应插图重命名
    3. 识别所有 txt 图片的 OCR，并按缩进生成段落文本，保存为 村名_数字.txt
"""
import os
import re
import shutil
import glob
import json
from pathlib import Path

# 导入自定义工具模块
from util.ocr_utils import ocr_image_to_text, ocr_image_to_json
from util.txt_extractor import extract_text_from_ocr_json

# ==================== 辅助函数 ====================
def sanitize_filename(name):
    """移除或替换文件名中的非法字符 (Windows/Linux)"""
    illegal_chars = r'[\\/*?:"<>|]'
    name = re.sub(illegal_chars, '_', name)
    name = re.sub(r'[\x00-\x1f]', '', name)
    name = name.strip()
    if not name:
        name = "未命名"
    return name

def parse_numeric_suffix(filename, pattern=r'_txt_(\d+)\.png$'):
    """从文件名如 Page_022_txt_1.png 中提取数字后缀（如 1）"""
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None

# ==================== 主处理函数 ====================
def process_cropped_data(
    input_root,           # Images_village_cropped 目录路径
    output_root,          # 各村OCR结果 目录路径
    temp_json_dir,        # 存放 OCR 中间 JSON 的临时目录（所有图片共用）
    device="cpu",         # "cpu" 或 "gpu"
    indent_threshold=80   # 缩进判断阈值（像素）
):
    """
    扫描 input_root 下所有以 _title 结尾的文件夹，依次处理。
    """
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(temp_json_dir, exist_ok=True)

    # 找出所有 title 文件夹（如 Page_022_title）
    title_folders = [
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d)) and '_title' in d
    ]
    if not title_folders:
        print(f"在 {input_root} 下未找到任何 title 文件夹")
        return

    print(f"共发现 {len(title_folders)} 个 title 文件夹")
    for folder in title_folders:
        folder_path = os.path.join(input_root, folder)
        print(f"\n正在处理: {folder_path}")

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

        # 使用稳定的 ocr_image_to_text 函数
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

        village_output_dir = os.path.join(output_root, village_name)
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

                # 识别 caption 文字作为插图名
                caption_text = ocr_image_to_text(
                    caption_path,
                    device=device,
                    temp_json_dir=os.path.join(temp_json_dir, "title_caption")
                )
                if not caption_text:
                    caption_text = "无标题插图"
                caption_text = sanitize_filename(caption_text)

                if os.path.exists(img_path):
                    dest_path = os.path.join(village_output_dir, f"{caption_text}.png")
                    shutil.copy2(img_path, dest_path)
                    print(f"    ✓ 插图已保存: {dest_path}")
                else:
                    print(f"    警告: 插图文件不存在 {img_path}")

        # ---------- 3. 处理所有 txt 图片（保留段落缩进）----------
        txt_images = sorted(
            glob.glob(os.path.join(folder_path, "*txt_*.png")),
            key=lambda x: int(parse_numeric_suffix(x) or 0)
        )
        print(f"  发现 {len(txt_images)} 个 txt 图片")

        for txt_img_path in txt_images:
            suffix_num = parse_numeric_suffix(os.path.basename(txt_img_path))
            if suffix_num is None:
                continue

            # 1) 保存 OCR 的 JSON 到临时目录（txt专用子目录）
            json_path = ocr_image_to_json(
                txt_img_path,
                os.path.join(temp_json_dir, "txt"),
                device=device
            )
            # 2) 从 JSON 中提取带缩进的文本
            formatted_text = extract_text_from_ocr_json(json_path, indent_threshold)
            # 3) 保存为 村名_数字.txt
            txt_out_name = f"{village_name}_{suffix_num}.txt"
            txt_out_path = os.path.join(village_output_dir, txt_out_name)
            with open(txt_out_path, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
            print(f"    ✓ 文本已保存: {txt_out_path}")

    print("\n所有处理完成！")

# ==================== 主入口 ====================
if __name__ == "__main__":
    # ===== 配置参数（请根据实际路径修改）=====
    INPUT_ROOT = r"Test\Images_village_cropped"          # crop_by_yolo_with_metadata 输出目录
    OUTPUT_ROOT = r"Test\各村OCR结果"                     # 最终输出目录
    TEMP_JSON_DIR = r"Test\OCR_json_Results"               # 存放所有中间 JSON 的临时目录
    DEVICE = "cpu"                                  # 可选 "cpu" 或 "gpu"
    INDENT_THRESHOLD = 80                           # 缩进判断阈值（像素）

    process_cropped_data(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        temp_json_dir=TEMP_JSON_DIR,
        device=DEVICE,
        indent_threshold=INDENT_THRESHOLD
    )