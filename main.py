"""
全粤村情 PDF → TXT 自动化处理入口
================================

基于 PaddleOCR + YOLO 的双栏 PDF 正文提取系统一键流水线。
用户仅需配置输入pdf路径 INPUT_PDF_PATH 即可完成全部处理。

流水线步骤：
  1. PDF → 高清 PNG 图片（300 DPI）
  2. YOLO 版面检测（标题/正文/插图/图注）
  3. 按村落标题智能裁剪归档（跨页归属 + 图注配对）
  4. PaddleOCR 批量文字识别
  5. 带缩进文本合并 + 按村落输出

Author: CaiHongJian
Date:   2026-07-28
"""

import os
import sys
import re
import json
import glob
import shutil
from pathlib import Path

from tqdm import tqdm
import fitz

# ========== 用户仅需修改以下两项 ==========
INPUT_PDF_PATH = r"D:/Download/282.江门市开平市卷（一）.pdf"          # 输入PDF路径
FINAL_OUTPUT_DIR = r"data/Final_output"        # 最终输出目录（默认，可以不改）
# ==========================================

# 技术参数（无需频繁修改）
PDF_DPI = 300                  # PDF转图分辨率；300DPI兼顾清晰度与性能，过低会导致OCR精度下降
YOLO_DEVICE = "auto"           # auto/gpu/cpu；auto自动检测CUDA可用性
YOLO_CONF_THRESH = 0.25        # YOLO置信度阈值；低于此值的目标被过滤，过高会漏检
YOLO_IOU_THRESH = 0.45         # NMS IOU阈值；控制重叠框合并强度
OCR_USE_GPU = False             # PaddleOCR是否使用GPU；False则强制CPU
OCR_MAX_BATCH_SIZE = 8         # OCR批处理大小；显存不足时可降至4或2
NUM_WORKERS = 4                # 并行worker数；CPU核心数的一半通常较优

# 模型与路径配置
MODEL_TEXT_PATH = r"models\Model_xhao\detect_text.pt"
MODEL_LAYOUT_PATH = r"models\Model_xhao\detect_layout.pt"
CLASSES_FILE = r"data\Dataset_village\classes.txt"

# 中间产物目录（自动管理，无需修改）
TEMP_DATA_DIR = r"data/Temp_data"
IMAGES_PDF_DIR = os.path.join(TEMP_DATA_DIR, "images_PDF")
IMAGES_CROPPED_DIR = os.path.join(TEMP_DATA_DIR, "images_cropped_villages")
OCR_JSON_DIR = os.path.join(TEMP_DATA_DIR, "ocr_json_results")


def print_environment_info():
    """
    打印运行环境信息，包括Python版本、各依赖库版本、CUDA状态、路径配置等。
    在程序启动时第一时间调用，帮助用户确认环境是否正确配置。
    """
    print("=" * 48)
    print("  全粤村情 PDF → TXT 自动化处理系统")
    print("=" * 48)

    info = {}

    info["Python"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    try:
        import torch
        info["PyTorch"] = torch.__version__
        cuda_available = torch.cuda.is_available()
        info["CUDA 可用"] = str(cuda_available)
        if cuda_available:
            info["当前设备"] = "GPU (CUDA)"
        else:
            info["当前设备"] = "CPU"
    except ImportError:
        info["PyTorch"] = "未安装"
        info["CUDA 可用"] = "未知"
        info["当前设备"] = "CPU (fallback)"

    try:
        import paddle
        info["PaddlePaddle"] = paddle.__version__
    except ImportError:
        info["PaddlePaddle"] = "未安装"

    try:
        from paddleocr import __version__ as paddleocr_version
        info["PaddleOCR"] = paddleocr_version
    except ImportError:
        try:
            import paddleocr
            info["PaddleOCR"] = getattr(paddleocr, "__version__", "未知")
        except ImportError:
            info["PaddleOCR"] = "未安装"

    try:
        import cv2
        info["OpenCV"] = cv2.__version__
    except ImportError:
        info["OpenCV"] = "未安装"

    try:
        info["PyMuPDF"] = fitz.__version__ if hasattr(fitz, '__version__') else str(fitz.version)
    except Exception:
        info["PyMuPDF"] = "未知"

    info["输入 PDF"] = INPUT_PDF_PATH
    info["输出目录"] = FINAL_OUTPUT_DIR

    print(">>> 环境信息")
    for key, val in info.items():
        print(f"  {key:<16s}: {val}")
    print("=" * 48)
    print()


def stage1_pdf_to_images(pdf_path, output_dir, dpi=300):
    """
    阶段1：将PDF文档批量转换为高清PNG图片

    参数:
        pdf_path:   输入PDF文件路径
        output_dir: 图片输出目录（data/Temp_data/images_PDF/）
        dpi:        输出分辨率，默认300；过低影响OCR精度，过高增加内存占用

    输出:
        Page_001.png, Page_002.png, ...

    依赖:
        PyMuPDF (fitz)
    """
    print("📄 Stage 1: PDF → 高清 PNG 图片")
    print(f"   输入: {pdf_path}")
    print(f"   输出: {output_dir}")
    print(f"   DPI:  {dpi}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    doc = fitz.open(pdf_path)
    page_count = doc.page_count

    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    with tqdm(total=page_count, desc="📄 PDF转图", position=1, leave=False,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        for page_num in range(page_count):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=mat)
            filename = f"Page_{page_num + 1:03d}.png"
            output_path = output_dir / filename
            pix.save(str(output_path))
            pbar.update(1)

    doc.close()

    print(f"   ✅ Stage 1 完成：{page_count} 页 → {output_dir}")
    print()
    return page_count


def stage2_yolo_detect(images_dir, output_dir, device="auto", conf_thresh=0.25, iou_thresh=0.45):
    """
    阶段2：使用YOLO双模型对PDF图片进行版面检测

    参数:
        images_dir:  输入图片目录（data/Temp_data/images_PDF/）
        output_dir:  输出目录，包含标注文件与可视化结果
        device:      推理设备，auto/gpu/cpu
        conf_thresh: 置信度阈值，低于此值的目标被过滤
        iou_thresh:  NMS IOU阈值，控制重叠框合并强度

    输出:
        labels/*.txt (YOLO标注)
        visual/*.jpg (可视化结果)
        classes.txt

    依赖:
        pipelines.Step1_YOLO_detect.detect_pdf_yolo_xhao (get_ordered_text_boxes, get_layout_boxes, draw_boxes_on_image)
    """
    print("🔍 Stage 2: YOLO 版面检测")

    images_dir = str(Path(images_dir))
    output_dir = str(Path(output_dir))

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")

    image_files = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp'):
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    total_images = len(image_files)

    if total_images == 0:
        raise RuntimeError(f"在 {images_dir} 中未找到任何图片")

    print(f"   发现 {total_images} 张图片")

    # 自动检测设备
    actual_device = device
    if device == "auto":
        try:
            import torch
            actual_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            actual_device = "cpu"
    print(f"   实际设备: {actual_device}")

    # 包装batch_detect_combined以添加进度条
    print(f"   加载模型: {MODEL_TEXT_PATH}, {MODEL_LAYOUT_PATH}")

    out_path = Path(output_dir)
    label_out = out_path / "labels"
    vis_out = out_path / "visual"
    label_out.mkdir(parents=True, exist_ok=True)
    vis_out.mkdir(parents=True, exist_ok=True)

    # 生成classes.txt
    classes_path = out_path / "classes.txt"
    classes_list = ['title', 'caption', 'txt_1', 'txt_2', 'img', 'txt_3', 'txt_4']
    with open(classes_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(classes_list))

    # 加载模型
    from ultralytics import YOLO
    model_text = YOLO(MODEL_TEXT_PATH)
    model_layout = YOLO(MODEL_LAYOUT_PATH)

    class_to_id = {name: idx for idx, name in enumerate(classes_list)}

    # 复用原脚本的辅助函数
    from pipelines.Step1_YOLO_detect.detect_pdf_yolo_xhao import (
        get_ordered_text_boxes, get_layout_boxes, draw_boxes_on_image
    )
    import cv2

    with tqdm(total=total_images, desc="🔍 YOLO检测", position=1, leave=False,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        for img_path in sorted(image_files):
            img_name = Path(img_path).stem

            img_out = out_path / "images"
            img_out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, img_out / Path(img_path).name)

            img = cv2.imread(img_path)
            if img is None:
                pbar.update(1)
                continue
            h, w = img.shape[:2]

            res_text = model_text(img_path, imgsz=640, conf=conf_thresh,
                                  device=actual_device, verbose=False)[0]
            res_layout = model_layout(img_path, imgsz=640, conf=conf_thresh,
                                      device=actual_device, verbose=False)[0]

            layout_boxes = get_layout_boxes(res_layout.boxes, model_layout.names)

            title_boxes = []
            for class_name, xywhn in layout_boxes:
                if class_name == 'title':
                    title_boxes.append(xywhn)

            text_boxes = get_ordered_text_boxes(
                res_text.boxes, model_text.names, w, h,
                title_boxes=title_boxes if title_boxes else None
            )

            all_boxes = text_boxes + layout_boxes

            txt_path = label_out / f"{img_name}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                for class_name, (xc, yc, ww, hh) in all_boxes:
                    cls_id = class_to_id[class_name]
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

            vis_img = img.copy()
            vis_img = draw_boxes_on_image(vis_img, all_boxes, model_text.names)
            vis_path = vis_out / f"{img_name}.jpg"
            cv2.imwrite(str(vis_path), vis_img)

            pbar.update(1)

    print(f"   ✅ Stage 2 完成：{total_images} 张图片 → {output_dir}")
    print()
    return total_images


def stage3_crop_by_yolo(images_dir, labels_dir, output_dir, classes_file):
    """
    阶段3：按YOLO标注智能裁剪图片并按村落归档

    参数:
        images_dir:    图片目录（data/Temp_data/images_PDF/）
        labels_dir:    标注目录（data/Temp_data/images_PDF/labels/）
        output_dir:    裁剪输出目录（data/Temp_data/images_cropped_villages/）
        classes_file:  类别定义文件路径

    输出:
        Page_022_title/ 等村落归档目录
        每个目录含裁剪图片和 img_caption_metadata.json

    依赖:
        pipelines.Step2_Crop_by_YOLO_Label.crop_by_yolo_with_metadata.process_folder
    """
    print("✂️  Stage 3: 智能裁剪归档")

    from pipelines.Step2_Crop_by_YOLO_Label.crop_by_yolo_with_metadata import process_folder

    images_dir = str(Path(images_dir))
    labels_dir = str(Path(labels_dir))
    output_dir = str(Path(output_dir))

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"类别文件不存在: {classes_file}")

    # 获取待处理图片数量用于进度条
    image_files = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp'):
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    total_images = len(image_files)

    if total_images == 0:
        raise RuntimeError(f"在 {images_dir} 中未找到任何图片")

    print(f"   发现 {total_images} 张图片待裁剪")

    # 直接调用原脚本的process_folder，它内部已有进度输出
    total_crops = process_folder(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        classes_file=classes_file
    )

    # 统计村落数
    title_folders = [d for d in os.listdir(output_dir)
                     if os.path.isdir(os.path.join(output_dir, d)) and 'title' in d]

    print(f"   ✅ Stage 3 完成：{total_images} 页 → {len(title_folders)} 个村落标题 → {output_dir}")
    print()
    return len(title_folders)


def stage4_ocr_recognition(cropped_dir, ocr_output_dir, use_gpu=True):
    """
    阶段4：对所有裁剪图片进行OCR识别，保存JSON中间结果

    参数:
        cropped_dir:   裁剪图片目录（data/Temp_data/images_cropped_villages/）
        ocr_output_dir: OCR JSON输出目录（data/Temp_data/ocr_json_results/）
        use_gpu:       是否使用GPU加速OCR

    输出:
        ocr_json_results/txt/*.json
        ocr_json_results/title_caption/*.json

    依赖:
        util.ocr_utils.ocr_image_to_json
    """
    print("📖 Stage 4: PaddleOCR 批量文字识别")

    from util.ocr_utils import ocr_image_to_json

    cropped_dir = str(Path(cropped_dir))
    ocr_output_dir = str(Path(ocr_output_dir))

    if not os.path.exists(cropped_dir):
        raise FileNotFoundError(f"裁剪目录不存在: {cropped_dir}")

    device = "gpu" if use_gpu else "cpu"

    # 收集所有需要OCR的图片
    all_images = []
    for root, dirs, files in os.walk(cropped_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                all_images.append(os.path.join(root, f))

    total_images = len(all_images)
    if total_images == 0:
        raise RuntimeError(f"在 {cropped_dir} 中未找到任何裁剪图片")

    print(f"   发现 {total_images} 张裁剪图片待OCR")
    print(f"   设备: {device}")

    # 分类：title/caption 图片存到 title_caption 子目录，txt 图片存到 txt 子目录
    title_caption_dir = os.path.join(ocr_output_dir, "title_caption")
    txt_dir = os.path.join(ocr_output_dir, "txt")
    os.makedirs(title_caption_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    with tqdm(total=total_images, desc="📖 OCR识别", position=1, leave=False,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        for img_path in all_images:
            img_name = Path(img_path).stem
            # 判断图片类型
            if '_title' in img_name or '_caption' in img_name:
                target_dir = title_caption_dir
            else:
                target_dir = txt_dir

            try:
                ocr_image_to_json(img_path, target_dir, device=device)
            except Exception as e:
                print(f"\n   ⚠ OCR 失败: {img_path} → {e}")

            pbar.update(1)

    # 统计OCR结果
    json_count = 0
    for root, dirs, files in os.walk(ocr_output_dir):
        json_count += sum(1 for f in files if f.endswith('.json'))

    print(f"   ✅ Stage 4 完成：{total_images} 张图片 → {json_count} 个JSON → {ocr_output_dir}")
    print()
    return total_images


def stage5_text_merge(ocr_json_dir, cropped_dir, final_output_dir, use_gpu=True):
    """
    阶段5：从OCR JSON结果提取文本，按村落合并，生成最终输出

    参数:
        ocr_json_dir:    OCR JSON结果目录（data/Temp_data/ocr_json_results/）
        cropped_dir:     裁剪图片目录（data/Temp_data/images_cropped_villages/）
        final_output_dir: 最终输出目录（data/Final_output/）
        use_gpu:         是否使用GPU（用于OCR村名和图注识别）

    输出:
        data/Final_output/各村OCR结果/1_大围村/
        data/Final_output/广州市从化区卷一_1-260.txt

    依赖:
        util.ocr_utils.ocr_image_to_text
        util.txt_extractor.extract_text_from_ocr_json
        util.txt_merger.parse_page_and_txt_num, merge_txt_segments
    """
    print("📝 Stage 5: 文本合并与最终输出")

    from util.ocr_utils import ocr_image_to_text
    from util.txt_extractor import extract_text_from_ocr_json
    from util.txt_merger import parse_page_and_txt_num, merge_txt_segments

    cropped_dir = str(Path(cropped_dir))
    final_output_dir = str(Path(final_output_dir))
    villages_output_dir = os.path.join(final_output_dir, "各村OCR结果")
    ocr_json_dir = str(Path(ocr_json_dir))

    Path(villages_output_dir).mkdir(parents=True, exist_ok=True)

    device = "gpu" if use_gpu else "cpu"

    # 辅助函数
    def sanitize_filename(name):
        illegal_chars = r'[\\/*?:"<>|]'
        name = re.sub(illegal_chars, '_', name)
        name = re.sub(r'[\x00-\x1f]', '', name)
        name = name.strip()
        if not name:
            name = "未命名"
        return name

    def clean_caption_text(text):
        if not text:
            return text
        if text[0] in ('O', 'o', '0'):
            text = text[1:].lstrip()
        return text

    # 找出所有title文件夹
    title_folders = sorted(
        [d for d in os.listdir(cropped_dir)
         if os.path.isdir(os.path.join(cropped_dir, d)) and '_title' in d],
        key=lambda x: int(re.search(r'Page_(\d+)', x).group(1)) if re.search(r'Page_(\d+)', x) else 0
    )

    if not title_folders:
        raise RuntimeError(f"在 {cropped_dir} 下未找到任何 title 文件夹")

    print(f"   发现 {len(title_folders)} 个村落标题")

    global_idx = 1
    all_village_texts = []

    with tqdm(total=len(title_folders), desc="📝 文本合并", position=1, leave=False,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        for folder in title_folders:
            folder_path = os.path.join(cropped_dir, folder)

            # 1. 识别title图片获取村名
            title_img_path = None
            for ext in ('*.png', '*.jpg', '*.jpeg'):
                candidates = glob.glob(os.path.join(folder_path, f"*title{ext[1:]}"))
                if candidates:
                    title_img_path = candidates[0]
                    break

            if not title_img_path:
                pbar.update(1)
                continue

            village_name = ocr_image_to_text(
                title_img_path,
                device=device,
                temp_json_dir=os.path.join(ocr_json_dir, "title_caption")
            )
            if not village_name:
                village_name = folder.replace('_title', '')
            village_name = sanitize_filename(village_name)

            numbered_folder_name = f"{global_idx}_{village_name}"
            village_output_dir = os.path.join(villages_output_dir, numbered_folder_name)
            os.makedirs(village_output_dir, exist_ok=True)

            # 2. 处理插图（利用img_caption_metadata.json）
            metadata_path = os.path.join(folder_path, "img_caption_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                pairs = metadata.get("pairs", [])

                for pair in pairs:
                    img_file = pair.get("img_filename")
                    caption_file = pair.get("caption_filename")
                    if not img_file or not caption_file:
                        continue
                    img_path = os.path.join(folder_path, img_file)
                    caption_path = os.path.join(folder_path, caption_file)

                    if not os.path.exists(caption_path):
                        continue

                    caption_text = ocr_image_to_text(
                        caption_path,
                        device=device,
                        temp_json_dir=os.path.join(ocr_json_dir, "title_caption")
                    )
                    if not caption_text:
                        caption_text = "无标题插图"
                    caption_text = clean_caption_text(caption_text)
                    caption_text = sanitize_filename(caption_text)

                    if os.path.exists(img_path):
                        dest_path = os.path.join(village_output_dir, f"{caption_text}.png")
                        counter = 1
                        while os.path.exists(dest_path):
                            dest_path = os.path.join(village_output_dir, f"{caption_text}_{counter}.png")
                            counter += 1
                        shutil.copy2(img_path, dest_path)

            # 3. 收集txt图片并合并文本
            txt_images = glob.glob(os.path.join(folder_path, "*txt_*.png"))
            if txt_images:
                segments = []
                for txt_img_path in txt_images:
                    filename = os.path.basename(txt_img_path)
                    page_num, txt_num = parse_page_and_txt_num(filename)
                    if page_num == 0:
                        continue

                    json_path = ocr_image_to_json(
                        txt_img_path,
                        os.path.join(ocr_json_dir, "txt"),
                        device=device
                    )
                    formatted_text = extract_text_from_ocr_json(json_path, indent_threshold=80)
                    if formatted_text:
                        segments.append({
                            'page_num': page_num,
                            'txt_num': txt_num,
                            'formatted_text': formatted_text
                        })

                if segments:
                    merged_content = merge_txt_segments(segments)
                    output_txt_name = f"{global_idx}_{village_name}.txt"
                    output_txt_path = os.path.join(village_output_dir, output_txt_name)
                    with open(output_txt_path, 'w', encoding='utf-8') as f:
                        f.write(merged_content)
                    all_village_texts.append(merged_content)

            global_idx += 1
            pbar.update(1)

    # 4. 合并所有村落文本为一个总文件
    if all_village_texts:
        combined_path = os.path.join(final_output_dir, "广州市从化区卷一_1-260.txt")
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(all_village_texts))

    print(f"   ✅ Stage 5 完成：{len(title_folders)} 个村落 → {villages_output_dir}")
    if all_village_texts:
        print(f"   ✅ 合并总文件 → {os.path.join(final_output_dir, '广州市从化区卷一_1-260.txt')}")
    print()
    return len(title_folders)


def main():
    """
    主入口：编排5阶段自动化流水线

    流程:
        1. 环境信息打印
        2. PDF → PNG (Stage 1)
        3. YOLO 版面检测 (Stage 2)
        4. 智能裁剪归档 (Stage 3)
        5. OCR 批量识别 (Stage 4)
        6. 文本合并输出 (Stage 5)

    异常处理:
        每个阶段检查前置依赖，遇异常打印清晰错误并退出
    """
    # 打印环境信息
    print_environment_info()

    # 参数验证
    if not os.path.exists(INPUT_PDF_PATH):
        print(f"❌ 错误：输入PDF不存在: {INPUT_PDF_PATH}")
        print("   请在 main.py 顶部修改 INPUT_PDF_PATH 为正确路径")
        sys.exit(1)

    if INPUT_PDF_PATH == r"path/to/input.pdf":
        print("❌ 错误：INPUT_PDF_PATH 仍为默认占位路径")
        print("   请在 main.py 顶部修改 INPUT_PDF_PATH 为实际PDF文件路径")
        sys.exit(1)

    # 创建目录
    Path(TEMP_DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(IMAGES_PDF_DIR).mkdir(parents=True, exist_ok=True)
    Path(IMAGES_CROPPED_DIR).mkdir(parents=True, exist_ok=True)
    Path(OCR_JSON_DIR).mkdir(parents=True, exist_ok=True)
    Path(FINAL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 计算实际OCR设备
    ocr_device_flag = OCR_USE_GPU
    if ocr_device_flag:
        try:
            import torch
            if not torch.cuda.is_available():
                print("⚠️  GPU不可用，OCR自动切换为CPU模式")
                ocr_device_flag = False
        except ImportError:
            ocr_device_flag = False

    # 总进度条：5个阶段
    with tqdm(total=5, desc="🚀 总体进度", position=0,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
              ncols=60) as pbar_total:

        # ===== Stage 1: PDF转图 =====
        try:
            stage1_pdf_to_images(
                pdf_path=INPUT_PDF_PATH,
                output_dir=IMAGES_PDF_DIR,
                dpi=PDF_DPI
            )
        except Exception as e:
            print(f"❌ Stage 1 失败: {e}")
            sys.exit(1)
        pbar_total.update(1)

        # Stage 1 完成检查
        png_files = glob.glob(os.path.join(IMAGES_PDF_DIR, "*.png"))
        if not png_files:
            print("❌ Stage 1 输出异常：未找到任何PNG图片")
            sys.exit(1)

        # ===== Stage 2: YOLO检测 =====
        try:
            stage2_yolo_detect(
                images_dir=IMAGES_PDF_DIR,
                output_dir=IMAGES_PDF_DIR,
                device=YOLO_DEVICE,
                conf_thresh=YOLO_CONF_THRESH,
                iou_thresh=YOLO_IOU_THRESH
            )
        except Exception as e:
            print(f"❌ Stage 2 失败: {e}")
            sys.exit(1)
        pbar_total.update(1)

        # Stage 2 完成检查
        label_files = glob.glob(os.path.join(IMAGES_PDF_DIR, "labels", "*.txt"))
        if not label_files:
            print("❌ Stage 2 输出异常：未找到任何标注文件")
            sys.exit(1)

        # ===== Stage 3: 智能裁剪 =====
        try:
            stage3_crop_by_yolo(
                images_dir=IMAGES_PDF_DIR,
                labels_dir=os.path.join(IMAGES_PDF_DIR, "labels"),
                output_dir=IMAGES_CROPPED_DIR,
                classes_file=CLASSES_FILE
            )
        except Exception as e:
            print(f"❌ Stage 3 失败: {e}")
            sys.exit(1)
        pbar_total.update(1)

        # Stage 3 完成检查
        title_dirs = [d for d in os.listdir(IMAGES_CROPPED_DIR)
                      if os.path.isdir(os.path.join(IMAGES_CROPPED_DIR, d)) and 'title' in d]
        if not title_dirs:
            print("❌ Stage 3 输出异常：未找到任何村落归档目录")
            sys.exit(1)

        # ===== Stage 4: OCR识别 =====
        try:
            stage4_ocr_recognition(
                cropped_dir=IMAGES_CROPPED_DIR,
                ocr_output_dir=OCR_JSON_DIR,
                use_gpu=ocr_device_flag
            )
        except Exception as e:
            print(f"❌ Stage 4 失败: {e}")
            sys.exit(1)
        pbar_total.update(1)

        # Stage 4 完成检查
        json_files = glob.glob(os.path.join(OCR_JSON_DIR, "**", "*.json"), recursive=True)
        if not json_files:
            print("❌ Stage 4 输出异常：未找到任何OCR JSON结果")
            sys.exit(1)

        # ===== Stage 5: 文本合并 =====
        try:
            stage5_text_merge(
                ocr_json_dir=OCR_JSON_DIR,
                cropped_dir=IMAGES_CROPPED_DIR,
                final_output_dir=FINAL_OUTPUT_DIR,
                use_gpu=ocr_device_flag
            )
        except Exception as e:
            print(f"❌ Stage 5 失败: {e}")
            sys.exit(1)
        pbar_total.update(1)

    # 最终统计
    print()
    print("=" * 48)
    print("  🎉 全流程处理完成！")
    print("=" * 48)

    # 统计最终产物
    villages_dir = os.path.join(FINAL_OUTPUT_DIR, "各村OCR结果")
    if os.path.exists(villages_dir):
        village_count = len([d for d in os.listdir(villages_dir)
                             if os.path.isdir(os.path.join(villages_dir, d))])
        print(f"  村落数量: {village_count}")

    combined_file = os.path.join(FINAL_OUTPUT_DIR, "广州市从化区卷一_1-260.txt")
    if os.path.exists(combined_file):
        file_size = os.path.getsize(combined_file)
        print(f"  总文本大小: {file_size:,} bytes")

    print(f"  输出目录: {os.path.abspath(FINAL_OUTPUT_DIR)}")
    print("=" * 48)


if __name__ == "__main__":
    main()