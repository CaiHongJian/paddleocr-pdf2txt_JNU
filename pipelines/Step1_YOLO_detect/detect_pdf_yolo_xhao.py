"""
Author: 熊浩
Date: 2026-05-10
Description: 整合正文模型和布局模型，按固定类别顺序输出 YOLO 标注文件，并生成可视化框选结果。

Modified by: 蔡鸿键, 2026-06-29, 修复正文框排序逻辑，改为按左上角坐标全局从上到下、从左到右排序，更符合阅读顺序
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


def get_ordered_text_boxes(boxes, model_names, img_width):
    """
    从正文模型的检测结果中提取所有文本框，按左上角坐标排序后分配类别名。
    
    [修改人: 蔡鸿键, 修改日期: 2026-06-29]
    修改内容：排序方式改为根据文本框左上角坐标 (x1, y1)，
    先按 y 坐标从上到下排序（y1 小的优先），再按 x 坐标从左到右排序（x1 小的优先），
    使正文块编号完全遵循自然阅读顺序。
    
    返回列表，每个元素为 (class_name, xywhn)，
    其中 class_name 为 'txt_1', 'txt_2', 'txt_3', 'txt_4'（最多4个）。
    """
    detections = []
    for box in boxes:
        cls = int(box.cls)
        cls_name = model_names[cls]
        # 只保留正文模型的文本框（通常类别名含 left 或 right）
        if 'left' not in cls_name.lower() and 'right' not in cls_name.lower():
            continue
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        xywhn = box.xywhn[0].tolist()  # [x_center, y_center, width, height]
        detections.append({
            'xywhn': xywhn,
            'x1': x1,
            'y1': y1
        })

    # [修改人: 蔡鸿键, 2026-06-29] 按左上角坐标排序：先 y（行），再 x（列）
    detections.sort(key=lambda d: (d['y1'], d['x1']))

    # 分配类别名，最多到 txt_4
    results = []
    for idx, det in enumerate(detections, start=1):
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
    在图像上绘制边界框和类别标签。
    boxes: 列表，每个元素为 (class_name, (xc, yc, ww, hh))
    class_names: 类别名称字典（用于显示标签）
    """
    h, w = image.shape[:2]
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

        # 选择颜色（如果类别不在预定义中，使用白色）
        color = COLORS.get(class_name, (255, 255, 255))

        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 准备标签文本
        label = class_name
        # 计算文本大小以绘制背景矩形
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # 绘制文本背景
        cv2.rectangle(image, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
        # 绘制文本
        cv2.putText(image, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
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

        # 获取排序后的正文框（已分配 txt_N 名称）
        text_boxes = get_ordered_text_boxes(res_text.boxes, model_text.names, w)
        # 获取布局框
        layout_boxes = get_layout_boxes(res_layout.boxes, model_layout.names)

        # 合并所有框
        all_boxes = text_boxes + layout_boxes

        # 写入 YOLO 格式标注文件（类别ID + 归一化坐标）
        txt_path = label_out / f"{img_name}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            for class_name, (xc, yc, ww, hh) in all_boxes:
                cls_id = CLASS_TO_ID[class_name]
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

        print(f"  已保存标注: {txt_path} (共 {len(all_boxes)} 个目标)")

        # ---------- 可视化部分 ----------
        # 在图像副本上绘制所有框
        vis_img = img.copy()
        vis_img = draw_boxes_on_image(vis_img, all_boxes, model_text.names)  # model_text.names 参数仅用于兼容，实际未使用
        # 保存可视化图像
        vis_path = vis_out / f"{img_name}.jpg"   # 统一保存为 .jpg 格式
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
    INPUT_DIR = r"data\Temp_data\images_PDF"
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