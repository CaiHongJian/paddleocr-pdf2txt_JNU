"""
Author: 熊浩
Date: 2026-05-10
Description: 整合正文模型和布局模型，按固定类别顺序输出 YOLO 标注文件，并生成可视化框选结果。
"""

import os
import glob
import shutil
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# 固定的类别顺序（与提供的 classes.txt 一致）
CLASSES = ['title', 'caption', 'txt_1', 'txt_2', 'img', 'txt_3', 'txt_4']
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}

# 为可视化定义固定的颜色（BGR格式）
COLORS = {
    'title': (0, 255, 0),    # 绿色
    'caption': (255, 255, 0), # 青色
    'txt_1': (255, 0, 0),    # 蓝色
    'txt_2': (0, 255, 255),  # 黄色
    'img': (0, 0, 255),      # 红色
    'txt_3': (128, 0, 255),  # 紫红
    'txt_4': (255, 128, 0),  # 橙色
}

def get_ordered_text_boxes(boxes, model_names, img_width, img_height, title_boxes=None):
    """
    从正文模型的检测结果中提取左右栏文本框，按阅读顺序排序，
    并支持按 title 位置分段（title 上方一组，下方一组）。
    返回列表，每个元素为 (class_name, xywhn)，
    其中 class_name 为 'txt_1', 'txt_2', ... 按实际顺序。
    """
    detections = []
    for box in boxes:
        cls = int(box.cls)
        cls_name = model_names[cls]
        # 假设正文模型的类别名包含 'left' 或 'right'
        if 'left' not in cls_name.lower() and 'right' not in cls_name.lower():
            continue
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        is_left = center_x < img_width * 0.5
        xywhn = box.xywhn[0].tolist()  # [x_center, y_center, width, height]
        detections.append({
            'xywhn': xywhn,
            'center_y': center_y,
            'is_left': is_left
        })

    # 如果没有检测到正文框，直接返回空
    if not detections:
        return []

    # 按 title 位置分段（如果有 title）
    if title_boxes:
        # 计算所有 title 框的下边界（y2）的最大值（像素坐标）
        max_title_y2 = 0
        for (xc, yc, ww, hh) in title_boxes:
            # 归一化坐标转像素：y2 = (yc + hh/2) * img_height
            y2 = (yc + hh / 2) * img_height
            if y2 > max_title_y2:
                max_title_y2 = y2
        # 分组：center_y < max_title_y2 为上方组，否则为下方组
        above = [d for d in detections if d['center_y'] < max_title_y2]
        below = [d for d in detections if d['center_y'] >= max_title_y2]
        groups = [above, below]
    else:
        # 没有 title 时，所有正文为一组
        groups = [detections]

    # 定义组内排序函数（先左后右，按 y 排序）
    def sort_group(group):
        left = [d for d in group if d['is_left']]
        right = [d for d in group if not d['is_left']]
        left.sort(key=lambda x: x['center_y'])
        right.sort(key=lambda x: x['center_y'])
        return left + right

    # 对所有组依次排序并合并
    ordered = []
    for group in groups:
        ordered.extend(sort_group(group))

    # 分配类别名 txt_1, txt_2, ...（最多到 txt_4，但可扩展）
    results = []
    for idx, det in enumerate(ordered, start=1):
        txt_name = f'txt_{idx}'
        if txt_name not in CLASS_TO_ID:
            # 如果超过预定义的 txt_4，可以忽略或继续添加（但 classes.txt 中没有，建议忽略）
            print(f"警告: 检测到第 {idx} 个正文块，但 classes.txt 只定义到 txt_4，该块将被忽略")
            continue
        results.append((txt_name, det['xywhn']))
    return results

def get_layout_boxes(boxes, model_names):
    """
    从布局模型提取 title, caption, img 框。
    返回列表，每个元素为 (class_name, xywhn)
    """
    results = []
    for box in boxes:
        cls = int(box.cls)
        cls_name = model_names[cls]
        if cls_name in CLASSES:
            xywhn = box.xywhn[0].tolist()
            results.append((cls_name, xywhn))
    return results

def draw_boxes_on_image(image, boxes, class_names):
    """
    在图像上绘制边界框和类别标签（字体更大更清晰）。
    boxes: 列表，每个元素为 (class_name, (xc, yc, ww, hh))
    class_names: 类别名称字典（用于显示标签）
    """
    h, w = image.shape[:2]
    font_scale = 1.5  # 调大字体
    thickness = 3     # 加粗线条
    for class_name, (xc, yc, ww, hh) in boxes:
        # 归一化坐标转换为绝对坐标
        x_center = xc * w
        y_center = yc * h
        width = ww * w
        height = hh * h
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # 选择颜色
        color = COLORS.get(class_name, (255, 255, 255))

        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # 准备标签文本
        label = class_name
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        # 绘制文本背景
        cv2.rectangle(image, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
        # 绘制文本（黑色）
        cv2.putText(image, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    return image

def batch_detect_combined(
    model_text_path: str,
    model_layout_path: str,
    input_dir: str,
    output_dir: str,
    conf_threshold: float = 0.25,
    imgsz: int = 640,
    device: str = "cuda",
):
    """
    整合两个模型，输出固定类别顺序的 YOLO 标注文件，并生成可视化结果。
    """
    out_path = Path(output_dir)
    img_out = out_path / "images"
    label_out = out_path / "labels"
    vis_out = out_path / "visual"
    img_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    vis_out.mkdir(parents=True, exist_ok=True)

    # 生成固定的 classes.txt
    classes_path = out_path / "classes.txt"
    with open(classes_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(CLASSES))
    print(f"已生成类别文件: {classes_path}")

    # 加载模型
    print(f"加载正文模型: {model_text_path}")
    model_text = YOLO(model_text_path)
    print(f"正文模型类别: {model_text.names}")
    print(f"加载布局模型: {model_layout_path}")
    model_layout = YOLO(model_layout_path)
    print(f"布局模型类别: {model_layout.names}")

    # 获取所有图片
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    image_files = sorted(list(set(image_files)))
    total = len(image_files)
    if total == 0:
        print(f"错误: 在 {input_dir} 中未找到图片")
        return

    print(f"共找到 {total} 张图片，开始处理...")

    for idx, img_path in enumerate(image_files, 1):
        img_name = Path(img_path).stem
        print(f"[{idx}/{total}] 处理: {img_name}")

        # 复制图片到输出目录
        shutil.copy2(img_path, img_out / Path(img_path).name)

        # 读取图片尺寸（用于排序和可视化）
        img = cv2.imread(img_path)
        if img is None:
            print(f"  无法读取图片，跳过")
            continue
        h, w = img.shape[:2]

        # 预测
        res_text = model_text(img_path, imgsz=imgsz, conf=conf_threshold, device=device, verbose=False)[0]
        res_layout = model_layout(img_path, imgsz=imgsz, conf=conf_threshold, device=device, verbose=False)[0]

        # 获取布局框（包含 title, caption, img）
        layout_boxes = get_layout_boxes(res_layout.boxes, model_layout.names)

        # 提取 title 框的归一化坐标（用于分段）
        title_boxes = []
        for class_name, xywhn in layout_boxes:
            if class_name == 'title':
                title_boxes.append(xywhn)

        # 获取排序后的正文框（传入 title_boxes 进行分段）
        text_boxes = get_ordered_text_boxes(
            res_text.boxes,
            model_text.names,
            w,
            h,
            title_boxes=title_boxes if title_boxes else None
        )

        # 合并所有框（layout_boxes 包含 title/caption/img，注意不要重复添加 title，因为已有）
        # 我们仍需保留 caption 和 img，但 title 已作为分段依据，但在最终标注中是否保留 title？
        # 原逻辑中 layout_boxes 会全部加入，包括 title，但 title 已经在 layout_boxes 中，我们应保留所有布局框（包括 title），
        # 因为 CLASSES 中包含 title，且输出需要 title 框。所以还是全部合并，但注意不要重复。
        # 我们使用 layout_boxes（包含所有布局框）加上 text_boxes。
        all_boxes = text_boxes + layout_boxes

        # 写入 YOLO 格式标注文件（类别ID + 归一化坐标）
        txt_path = label_out / f"{img_name}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            for class_name, (xc, yc, ww, hh) in all_boxes:
                cls_id = CLASS_TO_ID[class_name]
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

        print(f"  已保存标注: {txt_path} (共 {len(all_boxes)} 个目标)")

        # ---------- 可视化部分 ----------
        vis_img = img.copy()
        vis_img = draw_boxes_on_image(vis_img, all_boxes, model_text.names)
        vis_path = vis_out / f"{img_name}.jpg"
        cv2.imwrite(str(vis_path), vis_img)
        print(f"  已保存可视化: {vis_path}")

    print("\n处理完成！")
    print(f"图片目录: {img_out}")
    print(f"标签目录: {label_out}")
    print(f"可视化目录: {vis_out}")
    print(f"类别文件: {classes_path}")

if __name__ == '__main__':
    # ========== 配置参数 ==========
    MODEL_TEXT = r"models\Model_xhao\detect_text.pt"      # 正文模型（输出 left/right）
    MODEL_LAYOUT = r"models\Model_xhao\detect_layout.pt"  # 布局模型（输出 title/caption/img）
    # 输入目录：从此目录读取待检测的所有图片文件以及标签文件
    INPUT_DIR = r"data\Temp_data\images_PDF"
    # 输出目录：检测结果、标签文件、可视化图片
    OUTPUT_DIR = r"data\Temp_data\images_PDF"
    # ==============================

    batch_detect_combined(
        model_text_path=MODEL_TEXT,
        model_layout_path=MODEL_LAYOUT,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        conf_threshold=0.25,
        imgsz=640,
        device="cuda"
    )