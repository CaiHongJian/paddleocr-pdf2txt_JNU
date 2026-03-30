"""
Author: 熊浩
Date: 2026-3-29
Description:根据自训练模型，批量检测书页，按阅读顺序裁剪保存,命名格式：原图名_L_序号.jpg（左栏）, 原图名_R_序号.jpg（右栏）
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

def batch_detect_and_crop(
        model_path: str,  # 模型路径
        input_dir: str,  # 待检测图片文件夹
        output_dir: str,  # 结果保存文件夹
        conf: float = 0.5,  # 置信度阈值
        imgsz: int = 640,  # 输入尺寸
        save_vis: bool = True,  # 是否保存可视化图片
        jpeg_quality: int = 95,  # JPEG保存质量
):

    # 加载模型
    print(f"正在加载模型: {model_path}")
    model = YOLO(model_path)
    print(f"模型类别: {model.names}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vis_dir = output_path / "visualized"  # 画框后的图片
    txt_dir = output_path / "labels"  # YOLO格式坐标
    crop_dir = output_path / "crops"  # 裁剪的正文区域

    if save_vis:
        vis_dir.mkdir(exist_ok=True)
    txt_dir.mkdir(exist_ok=True)
    crop_dir.mkdir(exist_ok=True)

    # 获取所有 PNG 图片
    input_path = Path(input_dir)
    images = sorted(input_path.glob("*.png"))

    total = len(images)
    if total == 0:
        print(f"错误: 在 {input_dir} 中没有找到 PNG 图片")
        return

    print(f"\n找到 {total} 张待检测图片")
    print(f"输出目录: {output_path.absolute()}")
    print(f"置信度阈值: {conf}")
    print(f"JPEG质量: {jpeg_quality}")
    print("=" * 60)

    # 统计
    stats = {
        "total": total,
        "detected": 0,
        "total_crops": 0,
        "by_class": {}
    }

    # 批量检测
    for idx, img_path in enumerate(images, 1):
        print(f"\n[{idx}/{total}] 处理: {img_path.name}")

        # 读取原图（用于裁剪）
        orig_img = cv2.imread(str(img_path))
        if orig_img is None:
            print(f"   无法读取图片")
            continue

        img_height, img_width = orig_img.shape[:2]

        # 执行检测
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf,
            verbose=False,
        )[0]

        boxes = results.boxes
        num_boxes = len(boxes)

        if num_boxes == 0:
            print(f"   未检测到目标")
            (txt_dir / f"{img_path.stem}.txt").write_text("")
            continue

        # 提取所有检测框信息
        detections = []
        for box in boxes:
            cls = int(box.cls)
            cls_name = model.names[cls]
            conf_score = float(box.conf)

            # 获取像素坐标 (x1, y1, x2, y2)
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            # 归一化坐标
            xywhn = box.xywhn[0].tolist()

            # 判断左右（以图片中线为界）
            center_x = (x1 + x2) / 2
            is_left = center_x < img_width * 0.5

            detections.append({
                "cls": cls,
                "cls_name": cls_name,
                "conf": conf_score,
                "xyxy": (x1, y1, x2, y2),
                "xywhn": xywhn,
                "center_y": (y1 + y2) / 2,
                "is_left": is_left,  # True=左栏, False=右栏
            })

            stats["by_class"][cls_name] = stats["by_class"].get(cls_name, 0) + 1

        # 分离左右栏
        left_detections = [d for d in detections if d["is_left"]]
        right_detections = [d for d in detections if not d["is_left"]]

        # 分别按 y 坐标排序（从上到下）
        left_detections.sort(key=lambda x: x["center_y"])
        right_detections.sort(key=lambda x: x["center_y"])

        # 合并：左栏在前，右栏在后
        sorted_detections = left_detections + right_detections

        print(f"  ✓ 检测到 {num_boxes} 个目标")
        print(f"    左栏: {len(left_detections)} 个, 右栏: {len(right_detections)} 个")

        # 保存可视化图片
        if save_vis:
            vis_path = vis_dir / img_path.name
            results.save(filename=str(vis_path))

        # 保存标签文件（按排序后的顺序）
        txt_path = txt_dir / f"{img_path.stem}.txt"
        with open(txt_path, 'w') as f:
            for det in sorted_detections:
                c, x, y, w, h = det["cls"], det["xywhn"][0], det["xywhn"][1], det["xywhn"][2], det["xywhn"][3]
                f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {det['conf']:.6f}\n")

        # 裁剪并保存（新命名格式：原图名_L/R_序号.jpg）
        img_crops_dir = crop_dir / img_path.stem
        img_crops_dir.mkdir(exist_ok=True)

        # 处理左栏
        for i, det in enumerate(left_detections, 1):
            x1, y1, x2, y2 = det["xyxy"]

            # 边界检查
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)

            crop = orig_img[y1:y2, x1:x2]
            if crop.size == 0:
                print(f"    ✗ 左栏{i}裁剪区域为空")
                continue

            # 命名：page_0022_L_1.jpg
            crop_path = img_crops_dir / f"{img_path.stem}_L_{i}.jpg"
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            print(f"    保存: {crop_path.name} ({det['cls_name']}, {x2 - x1}x{y2 - y1})")

        # 处理右栏
        for i, det in enumerate(right_detections, 1):
            x1, y1, x2, y2 = det["xyxy"]

            # 边界检查
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)

            crop = orig_img[y1:y2, x1:x2]
            if crop.size == 0:
                print(f"    ✗ 右栏{i}裁剪区域为空")
                continue

            # 命名：page_0022_R_1.jpg
            crop_path = img_crops_dir / f"{img_path.stem}_R_{i}.jpg"
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            print(f"    保存: {crop_path.name} ({det['cls_name']}, {x2 - x1}x{y2 - y1})")

        stats["detected"] += 1
        stats["total_crops"] += len(sorted_detections)

    # 最终统计
    print("\n" + "=" * 60)
    print("检测完成！统计报告:")
    print(f"  总图片数: {stats['total']}")
    print(f"  有目标的图片: {stats['detected']} ({stats['detected'] / stats['total'] * 100:.1f}%)")
    print(f"  总裁剪数: {stats['total_crops']}")
    print(f"\n  按类别统计:")
    for cls_name, count in sorted(stats["by_class"].items()):
        print(f"    {cls_name}: {count} 个")

    print(f"\n结果保存位置:")
    if save_vis:
        print(f"  可视化图片: {vis_dir}")
    print(f"  坐标标签: {txt_dir}")
    print(f"  裁剪正文: {crop_dir}")
    print(f"\n裁剪命名示例:")
    print(f"  page_0022.png (左栏2个, 右栏2个):")
    print(f"    ├── page_0022_L_1.jpg  (左栏第1个，从上到下)")
    print(f"    ├── page_0022_L_2.jpg  (左栏第2个)")
    print(f"    ├── page_0022_R_1.jpg  (右栏第1个，从上到下)")
    print(f"    └── page_0022_R_2.jpg  (右栏第2个)")


if __name__ == '__main__':
    # ========== 路径配置 ==========
    MODEL_PATH = r"Models_text\textV1.pt"
    INPUT_FOLDER = r"BookImages"
    OUTPUT_FOLDER = r"output"
    # ==============================

    batch_detect_and_crop(
        model_path=MODEL_PATH,
        input_dir=INPUT_FOLDER,
        output_dir=OUTPUT_FOLDER,
        conf=0.5,  # 置信度阈值
        imgsz=640,  # 输入尺寸
        save_vis=True,  # 保存画框后的可视化图
        jpeg_quality=95,  # JPEG质量
    )