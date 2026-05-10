"""
YOLO 格式数据集图片分割脚本（增强版：按 title 分类归档）
功能：根据 YOLO 标注坐标分割图片，并按 title 归属关系分类归档
"""

import os
import cv2
import glob
import json
from pathlib import Path


def load_classes(classes_file):
    """
    加载类别名称文件
    """
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    print(f"加载类别: {classes}")
    return classes


def parse_yolo_label(txt_path, img_width, img_height):
    """
    解析 YOLO 格式标注文件，返回带绝对坐标的标注框列表
    格式: class_id x_center y_center width height
    """
    boxes = []
    
    if not os.path.exists(txt_path):
        print(f"警告: 标注文件不存在: {txt_path}")
        return boxes
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                print(f"警告: 格式错误的行: {line}")
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # 转换为绝对像素坐标
            x_center_abs = x_center * img_width
            y_center_abs = y_center * img_height
            width_abs = width * img_width
            height_abs = height * img_height
            
            # 计算左上角和右下角坐标
            x1 = int(x_center_abs - width_abs / 2)
            y1 = int(y_center_abs - height_abs / 2)
            x2 = int(x_center_abs + width_abs / 2)
            y2 = int(y_center_abs + height_abs / 2)
            
            # 边界检查
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            
            boxes.append({
                'class_id': class_id,
                'class_name': None,  # 稍后填充
                'x_center_abs': x_center_abs,
                'y_center_abs': y_center_abs,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })
    
    return boxes


def get_title_class_id(classes):
    """
    获取 title 类别的 class_id
    假设 classes.txt 中 title 的类别名包含 'title' 字样
    """
    for idx, name in enumerate(classes):
        if 'title' in name.lower():
            return idx
    # 如果没找到，尝试找最后一个类别（通常 title 是最后标注的）
    print("警告: 未找到包含 'title' 的类别，将使用最后一个类别作为 title")
    return len(classes) - 1 if classes else -1


def collect_all_titles(image_files, labels_dir, classes, title_class_id):
    """
    【第一遍扫描】收集所有页面中的所有 title 信息
    
    返回: 按页面顺序排列的 title 列表，每个元素包含：
        - page_name: 页面名称（如 Page_022）
        - title_name: title 的完整名称（如 Page_022_title）
        - y_center: title 在页面中的 y 中心坐标（绝对像素）
        - page_index: 页面在总序列中的索引
    """
    all_titles = []
    
    for page_idx, img_path in enumerate(image_files):
        img_name = Path(img_path).stem  # 如 Page_022
        
        # 读取图片获取尺寸
        image = cv2.imread(img_path)
        if image is None:
            continue
        img_height, img_width = image.shape[:2]
        
        # 解析该页面的标注
        txt_path = os.path.join(labels_dir, f"{img_name}.txt")
        boxes = parse_yolo_label(txt_path, img_width, img_height)
        
        # 填充类别名称
        for box in boxes:
            if 0 <= box['class_id'] < len(classes):
                box['class_name'] = classes[box['class_id']]
        
        # 收集该页面中的所有 title
        page_titles = []
        for box in boxes:
            if box['class_id'] == title_class_id:
                # title 的名称格式为 Page_xxx_title
                title_name = f"{img_name}_title"
                page_titles.append({
                    'page_name': img_name,
                    'title_name': title_name,
                    'y_center': box['y_center_abs'],
                    'page_index': page_idx,
                    'box': box
                })
        
        # 按 y_center 排序（从上到下）
        page_titles.sort(key=lambda t: t['y_center'])
        all_titles.extend(page_titles)
    
    print(f"\n【第一遍扫描完成】共发现 {len(all_titles)} 个 title")
    for t in all_titles:
        print(f"  - {t['title_name']} (第{t['page_index']}页, y_center={t['y_center']:.0f})")
    
    return all_titles


def find_title_for_box(box, current_page_idx, current_page_height, all_titles):
    """
    【核心判断逻辑】判断一个非 title 的标注框属于哪个 title
    
    规则：
    1. 在当前页面中，找到位于该 box 上方（y_center 更小）且最接近的 title
    2. 如果当前页面没有这样的 title，则往前查找最近的一个 title
    3. title 的作用域从其出现位置开始，直到下一个 title 出现前（跨页有效）
    
    参数:
        box: 当前待判断的标注框
        current_page_idx: 当前页面索引
        current_page_height: 当前页面高度
        all_titles: 所有收集到的 title 列表（已按页面和 y_center 排序）
    """
    box_y_center = box['y_center_abs']
    
    # 筛选出在当前 box 之前出现的所有 title
    # "之前"的定义：同一页面中 y_center 更小，或者是更早的页面
    candidate_titles = []
    for t in all_titles:
        if t['page_index'] < current_page_idx:
            # 更早页面的 title，都是候选
            candidate_titles.append(t)
        elif t['page_index'] == current_page_idx:
            # 同一页面，只有 y_center 在 box 上方的 title 才是候选
            if t['y_center'] < box_y_center:
                candidate_titles.append(t)
            else:
                # 由于 all_titles 已排序，遇到第一个 y_center >= box 的就可以停了
                # 但这里要考虑：如果当前页面有多个 title，box 在 titleA 和 titleB 之间
                # 则应该归属 titleA
                break
    
    if not candidate_titles:
        # 没有找到任何前置 title，使用第一个 title（兜底）
        if all_titles:
            return all_titles[0]['title_name']
        return None
    
    # 取最后一个候选 title（即最接近当前 box 的前一个 title）
    # 因为 candidate_titles 是按页面和 y_center 排序的
    assigned_title = candidate_titles[-1]['title_name']
    return assigned_title


def crop_and_save(image, box, class_name, output_dir, img_name, suffix=""):
    """
    裁剪指定区域并保存
    增加 suffix 参数用于处理同一类多个实例的情况
    """
    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
    
    # 检查裁剪区域是否有效
    if x2 <= x1 or y2 <= y1:
        print(f"  跳过无效区域: {img_name} -> {class_name} (w:{x2-x1}, h:{y2-y1})")
        return False
    
    # 裁剪图片
    cropped = image[y1:y2, x1:x2]
    
    # 生成文件名: Page_xxx_标签名.png，如有重复则加序号
    if suffix:
        output_name = f"{img_name}_{class_name}_{suffix}.png"
    else:
        output_name = f"{img_name}_{class_name}.png"
    output_path = os.path.join(output_dir, output_name)
    
    # 处理文件名冲突
    counter = 1
    base_output_path = output_path
    while os.path.exists(output_path):
        output_name = f"{img_name}_{class_name}_{counter}.png"
        output_path = os.path.join(output_dir, output_name)
        counter += 1
    
    # 保存
    success = cv2.imwrite(output_path, cropped)
    
    if success:
        print(f"  ✓ 已保存: {os.path.basename(output_path)} ({x2-x1}x{y2-y1})")
        return True
    else:
        print(f"  ✗ 保存失败: {os.path.basename(output_path)}")
        return False


def process_single_image(img_path, txt_path, classes, title_class_id, all_titles, page_idx, base_output_dir):
    """
    【第二遍处理】处理单张图片，按 title 归属分类保存
    
    参数:
        img_path: 图片路径
        txt_path: 标注文件路径
        classes: 类别列表
        title_class_id: title 的类别 ID
        all_titles: 所有收集到的 title 信息
        page_idx: 当前页面在所有图片中的索引
        base_output_dir: 基础输出目录（如 Images_village_cropped）
    """
    img_name = Path(img_path).stem  # 不含扩展名，如 Page_022
    
    # 读取图片
    image = cv2.imread(img_path)
    if image is None:
        print(f"✗ 无法读取图片: {img_path}")
        return 0
    
    img_height, img_width = image.shape[:2]
    print(f"\n处理: {img_name} ({img_width}x{img_height})")
    
    # 解析标注
    boxes = parse_yolo_label(txt_path, img_width, img_height)
    
    if not boxes:
        print(f"  未找到标注信息")
        return 0
    
    # 填充类别名称
    for box in boxes:
        if 0 <= box['class_id'] < len(classes):
            box['class_name'] = classes[box['class_id']]
    
    # 统计每个 title 文件夹下各类别的计数（用于处理重名）
    title_class_counters = {}
    
    # 裁剪并保存每个非 title 区域
    saved_count = 0
    
    # 先处理 title 本身（保存到对应的 title 文件夹中）
    for box in boxes:
        if box['class_id'] == title_class_id:
            # title 自身的图片保存到自己的文件夹
            title_name = f"{img_name}_title"
            title_output_dir = os.path.join(base_output_dir, title_name)
            Path(title_output_dir).mkdir(parents=True, exist_ok=True)
            
            if crop_and_save(image, box, box['class_name'], title_output_dir, img_name):
                saved_count += 1
    
    # 再处理非 title 区域，根据归属关系保存
    for box in boxes:
        if box['class_id'] == title_class_id:
            continue  # title 已处理
        
        # 【核心】判断该 box 属于哪个 title
        assigned_title = find_title_for_box(
            box, 
            page_idx, 
            img_height, 
            all_titles
        )
        
        if assigned_title is None:
            print(f"  警告: {img_name} 的 {box['class_name']} 无法找到归属 title，跳过")
            continue
        
        # 创建该 title 的输出目录
        title_output_dir = os.path.join(base_output_dir, assigned_title)
        Path(title_output_dir).mkdir(parents=True, exist_ok=True)
        
        # 生成唯一文件名（防止同一 title 下同类别的多个实例覆盖）
        class_name = box['class_name'] or f"class_{box['class_id']}"
        
        # 使用计数器确保文件名唯一
        key = f"{assigned_title}_{class_name}"
        if key not in title_class_counters:
            title_class_counters[key] = 0
        title_class_counters[key] += 1
        count = title_class_counters[key]
        
        suffix = f"{count}" if count > 1 else ""
        
        # 裁剪保存
        if crop_and_save(image, box, class_name, title_output_dir, img_name, suffix):
            saved_count += 1
    
    return saved_count


def process_folder(
    images_dir,
    labels_dir,
    output_dir,
    classes_file
):
    """
    处理指定文件夹内的所有图片，按 title 分类归档
    
    参数:
        images_dir: 图片文件夹路径
        labels_dir: 标注文件夹路径
        output_dir: 输出文件夹路径（每个 title 一个子文件夹）
        classes_file: classes.txt 路径
    """
    # 加载类别
    classes = load_classes(classes_file)
    
    # 获取 title 的类别 ID
    title_class_id = get_title_class_id(classes)
    print(f"title 类别 ID: {title_class_id} ({classes[title_class_id] if 0 <= title_class_id < len(classes) else '未知'})")
    
    # 创建基础输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片并按文件名排序（确保页面顺序正确）
    image_files = sorted(
        glob.glob(os.path.join(images_dir, '*.png')) +
        glob.glob(os.path.join(images_dir, '*.jpg')) +
        glob.glob(os.path.join(images_dir, '*.jpeg')) +
        glob.glob(os.path.join(images_dir, '*.bmp')) +
        glob.glob(os.path.join(images_dir, '*.webp'))
    )
    
    total_images = len(image_files)
    if total_images == 0:
        print("错误: 未找到任何图片文件")
        return 0
    
    print(f"\n{'='*60}")
    print(f"发现 {total_images} 张图片")
    print(f"{'='*60}")
    
    # ========================================
    # 【第一遍扫描】收集所有 title 信息
    # ========================================
    print("\n【阶段一】扫描所有页面，收集 title 坐标信息...")
    all_titles = collect_all_titles(image_files, labels_dir, classes, title_class_id)
    
    if not all_titles:
        print("错误: 未找到任何 title，无法进行分类归档")
        return 0
    
    # ========================================
    # 【第二遍处理】按 title 归属分割并保存
    # ========================================
    print(f"\n{'='*60}")
    print("【阶段二】按 title 归属关系分割图片并分类归档...")
    print(f"{'='*60}")
    
    total_crops = 0
    
    for idx, img_path in enumerate(image_files, 1):
        img_name = Path(img_path).stem
        txt_path = os.path.join(labels_dir, f"{img_name}.txt")
        
        # 处理单张图片
        crops = process_single_image(
            img_path=img_path,
            txt_path=txt_path,
            classes=classes,
            title_class_id=title_class_id,
            all_titles=all_titles,
            page_idx=idx - 1,  # 从0开始的索引
            base_output_dir=output_dir
        )
        total_crops += crops
        
        # 进度显示
        if idx % 10 == 0 or idx == total_images:
            print(f"\n进度: [{idx}/{total_images}] 已裁剪 {total_crops} 个区域")
    
    # ========================================
    # 输出统计信息
    # ========================================
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"处理图片: {total_images} 张")
    print(f"发现 title: {len(all_titles)} 个")
    print(f"裁剪区域: {total_crops} 个")
    print(f"输出目录: {output_dir}")
    
    # 列出每个 title 文件夹的内容
    print(f"\n【归档结果】")
    title_folders = sorted([d for d in os.listdir(output_dir) 
                           if os.path.isdir(os.path.join(output_dir, d)) and 'title' in d])
    for folder in title_folders:
        folder_path = os.path.join(output_dir, folder)
        files = os.listdir(folder_path)
        print(f"  📁 {folder}/ : {len(files)} 个文件")
    
    print(f"{'='*60}")
    
    return total_crops


def main():
    """
    主函数：配置参数并运行
    """
    # ========================================
    # 配置参数（根据你的实际情况修改）
    # ========================================
    
    # 图片文件夹路径
    IMAGES_DIR = r"Dataset_village\images\train"  # 或 images\val，或任意图片文件夹
    
    # 标注文件夹路径
    LABELS_DIR = r"Dataset_village\labels\train"  # 与图片对应的标注文件夹
    
    # 类别文件路径
    CLASSES_FILE = r"Dataset_village\classes.txt"
    
    # 统一输出目录（每个 title 一个子文件夹）
    OUTPUT_DIR = r"Images_village_cropped"
    
    # ========================================
    # 运行处理
    # ========================================
    
    process_folder(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        output_dir=OUTPUT_DIR,
        classes_file=CLASSES_FILE
    )


if __name__ == "__main__":
    main()