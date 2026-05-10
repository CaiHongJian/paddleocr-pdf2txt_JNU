"""
YOLO 格式数据集图片分割脚本（增强版：按 title 分类归档 + 记录 img/caption 元数据）
功能：
    1. 根据 YOLO 标注坐标分割图片，并按 title 归属关系分类归档
    2. 记录每个 img 和 caption 的 YOLO 坐标，依据“caption 在 img 下方”特性建立配对
    3. 为每个 title 文件夹生成 JSON 元数据文件，便于后续 OCR 识别 caption 并重命名 img
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


def get_class_id_by_keyword(classes, keyword):
    """
    根据关键词获取类别 ID（返回第一个匹配的，若无则返回 -1）
    """
    for idx, name in enumerate(classes):
        if keyword.lower() in name.lower():
            return idx
    return -1


def parse_yolo_label(txt_path, img_width, img_height):
    """
    解析 YOLO 格式标注文件，返回带绝对坐标的标注框列表
    格式: class_id x_center y_center width height
    同时保留原始 YOLO 相对坐标
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
            
            # 保存原始 YOLO 相对坐标
            yolo_coords = (x_center, y_center, width, height)
            
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
                'class_name': None,          # 稍后填充
                'yolo_coords': yolo_coords,  # 原始相对坐标
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
    """
    box_y_center = box['y_center_abs']
    
    # 筛选出在当前 box 之前出现的所有 title
    candidate_titles = []
    for t in all_titles:
        if t['page_index'] < current_page_idx:
            candidate_titles.append(t)
        elif t['page_index'] == current_page_idx:
            if t['y_center'] < box_y_center:
                candidate_titles.append(t)
            else:
                break
    
    if not candidate_titles:
        if all_titles:
            return all_titles[0]['title_name']
        return None
    
    # 取最后一个（即最接近的）候选 title
    return candidate_titles[-1]['title_name']


def crop_and_save(image, box, class_name, output_dir, img_name, suffix=""):
    """
    裁剪指定区域并保存，返回保存的文件名（不含路径）或 None
    """
    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
    
    if x2 <= x1 or y2 <= y1:
        print(f"  跳过无效区域: {img_name} -> {class_name} (w:{x2-x1}, h:{y2-y1})")
        return None
    
    cropped = image[y1:y2, x1:x2]
    
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
        if counter > 100:
            break
    
    success = cv2.imwrite(output_path, cropped)
    if success:
        print(f"  ✓ 已保存: {os.path.basename(output_path)} ({x2-x1}x{y2-y1})")
        return output_name
    else:
        print(f"  ✗ 保存失败: {os.path.basename(output_path)}")
        return None


def pair_img_caption(imgs, captions):
    """
    配对 img 和 caption。
    规则：每个 caption 匹配其上方最近的 img（y_center 小于 caption 且差值最小）。
    返回配对列表，每个元素为 (img_info, caption_info)，若某个 img 没有对应 caption，则 caption_info 为 None。
    """
    if not imgs:
        return []
    
    # 按 y_center 排序
    imgs_sorted = sorted(imgs, key=lambda x: x['y_center'])
    captions_sorted = sorted(captions, key=lambda x: x['y_center'])
    
    pairs = []
    used_captions = set()
    
    for img in imgs_sorted:
        best_caption = None
        best_dist = float('inf')
        for cap in captions_sorted:
            if id(cap) in used_captions:
                continue
            if cap['y_center'] > img['y_center']:  # caption 在 img 下方
                dist = cap['y_center'] - img['y_center']
                if dist < best_dist:
                    best_dist = dist
                    best_caption = cap
        if best_caption:
            used_captions.add(id(best_caption))
            pairs.append((img, best_caption))
        else:
            pairs.append((img, None))
    
    return pairs


def process_single_image(img_path, txt_path, classes, title_class_id,
                         img_class_id, caption_class_id,
                         all_titles, page_idx, base_output_dir, title_metadata):
    """
    【第二遍处理】处理单张图片：
        - 按 title 归属裁剪保存所有区域
        - 收集 img 和 caption 信息并建立配对，存入 title_metadata
    """
    img_name = Path(img_path).stem
    image = cv2.imread(img_path)
    if image is None:
        print(f"✗ 无法读取图片: {img_path}")
        return 0
    
    img_height, img_width = image.shape[:2]
    print(f"\n处理: {img_name} ({img_width}x{img_height})")
    
    boxes = parse_yolo_label(txt_path, img_width, img_height)
    if not boxes:
        print(f"  未找到标注信息")
        return 0
    
    # 填充类别名称
    for box in boxes:
        if 0 <= box['class_id'] < len(classes):
            box['class_name'] = classes[box['class_id']]
    
    # ========== 第一步：计算每个非 title 框的归属 title ==========
    # 同时按 title 分组收集 img 和 caption 的临时信息（尚未保存图片）
    # 结构: per_title[title_name] = {'imgs': [], 'captions': []}
    per_title = {}
    
    for box in boxes:
        if box['class_id'] == title_class_id:
            continue  # title 单独处理，不在此分组
        
        assigned_title = find_title_for_box(box, page_idx, img_height, all_titles)
        if assigned_title is None:
            print(f"  警告: {img_name} 的 {box['class_name']} 无法找到归属 title，跳过")
            continue
        
        # 准备该 title 的容器
        if assigned_title not in per_title:
            per_title[assigned_title] = {'imgs': [], 'captions': []}
        
        # 收集信息（待保存图片）
        info = {
            'box': box,
            'class_name': box['class_name'],
            'y_center': box['y_center_abs'],
            'yolo_coords': box['yolo_coords'],
            'abs_coords': (box['x1'], box['y1'], box['x2'], box['y2']),
            'saved_filename': None   # 稍后填充
        }
        
        if box['class_id'] == img_class_id:
            per_title[assigned_title]['imgs'].append(info)
        elif box['class_id'] == caption_class_id:
            per_title[assigned_title]['captions'].append(info)
        # 其他类别（如页眉、页脚等）不记录到元数据，但仍会裁剪保存（下面统一处理）
    
    # ========== 第二步：处理 title 自身（裁剪并保存） ==========
    for box in boxes:
        if box['class_id'] == title_class_id:
            title_name = f"{img_name}_title"
            title_output_dir = os.path.join(base_output_dir, title_name)
            Path(title_output_dir).mkdir(parents=True, exist_ok=True)
            crop_and_save(image, box, box['class_name'], title_output_dir, img_name)
    
    # ========== 第三步：统一裁剪并保存所有非 title 区域 ==========
    # 同时记录最终保存的文件名
    total_saved = 0
    # 用于同一 title 内同类多实例的计数器（防止覆盖）
    title_class_counters = {}
    
    # 先处理 img 和 caption 以外的类别（仅保存，不参与配对）
    for box in boxes:
        if box['class_id'] in (title_class_id, img_class_id, caption_class_id):
            continue
        assigned_title = find_title_for_box(box, page_idx, img_height, all_titles)
        if assigned_title is None:
            continue
        title_output_dir = os.path.join(base_output_dir, assigned_title)
        Path(title_output_dir).mkdir(parents=True, exist_ok=True)
        class_name = box['class_name'] or f"class_{box['class_id']}"
        key = f"{assigned_title}_{class_name}"
        title_class_counters[key] = title_class_counters.get(key, 0) + 1
        suffix = str(title_class_counters[key]) if title_class_counters[key] > 1 else ""
        filename = crop_and_save(image, box, class_name, title_output_dir, img_name, suffix)
        if filename:
            total_saved += 1
    
    # 处理 img 和 caption：保存并填充文件名到 info 中
    for title_name, groups in per_title.items():
        title_output_dir = os.path.join(base_output_dir, title_name)
        Path(title_output_dir).mkdir(parents=True, exist_ok=True)
        
        # 为 img 和 caption 分别分配唯一后缀（避免同类别多实例冲突）
        img_counters = {}
        cap_counters = {}
        
        for img_info in groups['imgs']:
            class_name = img_info['class_name']
            key = f"{title_name}_{class_name}"
            img_counters[key] = img_counters.get(key, 0) + 1
            suffix = str(img_counters[key]) if img_counters[key] > 1 else ""
            filename = crop_and_save(image, img_info['box'], class_name,
                                     title_output_dir, img_name, suffix)
            img_info['saved_filename'] = filename
        
        for cap_info in groups['captions']:
            class_name = cap_info['class_name']
            key = f"{title_name}_{class_name}"
            cap_counters[key] = cap_counters.get(key, 0) + 1
            suffix = str(cap_counters[key]) if cap_counters[key] > 1 else ""
            filename = crop_and_save(image, cap_info['box'], class_name,
                                     title_output_dir, img_name, suffix)
            cap_info['saved_filename'] = filename
        
        # 配对 img 和 caption
        pairs = pair_img_caption(groups['imgs'], groups['captions'])
        
        # 将配对信息存入全局 title_metadata
        if title_name not in title_metadata:
            title_metadata[title_name] = []
        
        for img_info, cap_info in pairs:
            record = {
                'img_filename': img_info['saved_filename'],
                'img_yolo_coords': list(img_info['yolo_coords']),     # [x_center, y_center, width, height]
                'img_abs_coords': {
                    'x1': img_info['abs_coords'][0],
                    'y1': img_info['abs_coords'][1],
                    'x2': img_info['abs_coords'][2],
                    'y2': img_info['abs_coords'][3]
                }
            }
            if cap_info:
                record['caption_filename'] = cap_info['saved_filename']
                record['caption_yolo_coords'] = list(cap_info['yolo_coords'])
                record['caption_abs_coords'] = {
                    'x1': cap_info['abs_coords'][0],
                    'y1': cap_info['abs_coords'][1],
                    'x2': cap_info['abs_coords'][2],
                    'y2': cap_info['abs_coords'][3]
                }
            else:
                record['caption_filename'] = None
                record['caption_yolo_coords'] = None
                record['caption_abs_coords'] = None
            
            title_metadata[title_name].append(record)
        
        total_saved += len(groups['imgs']) + len(groups['captions'])
    
    return total_saved


def process_folder(images_dir, labels_dir, output_dir, classes_file):
    """
    处理指定文件夹内的所有图片，按 title 分类归档，并生成每个 title 的元数据 JSON
    """
    # 加载类别
    classes = load_classes(classes_file)
    
    # 获取 title / img / caption 的类别 ID
    title_class_id = get_title_class_id(classes)
    img_class_id = get_class_id_by_keyword(classes, 'img')
    caption_class_id = get_class_id_by_keyword(classes, 'caption')
    
    print(f"title 类别 ID: {title_class_id} ({classes[title_class_id] if 0 <= title_class_id < len(classes) else '未知'})")
    print(f"img 类别 ID: {img_class_id} ({classes[img_class_id] if 0 <= img_class_id < len(classes) else '未找到'})")
    print(f"caption 类别 ID: {caption_class_id} ({classes[caption_class_id] if 0 <= caption_class_id < len(classes) else '未找到'})")
    
    if img_class_id == -1 or caption_class_id == -1:
        print("警告: 未找到 img 或 caption 类别，元数据记录将不完整")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片并排序
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
    
    # ===== 阶段一：收集所有 title =====
    print("\n【阶段一】扫描所有页面，收集 title 坐标信息...")
    all_titles = collect_all_titles(image_files, labels_dir, classes, title_class_id)
    if not all_titles:
        print("错误: 未找到任何 title，无法进行分类归档")
        return 0
    
    # ===== 阶段二：按 title 归属分割并收集元数据 =====
    print(f"\n{'='*60}")
    print("【阶段二】按 title 归属关系分割图片，同时收集 img/caption 元数据...")
    print(f"{'='*60}")
    
    # 全局元数据容器: { title_name: [ records ] }
    title_metadata = {}
    total_crops = 0
    
    for idx, img_path in enumerate(image_files, 1):
        img_name = Path(img_path).stem
        txt_path = os.path.join(labels_dir, f"{img_name}.txt")
        
        crops = process_single_image(
            img_path=img_path,
            txt_path=txt_path,
            classes=classes,
            title_class_id=title_class_id,
            img_class_id=img_class_id,
            caption_class_id=caption_class_id,
            all_titles=all_titles,
            page_idx=idx - 1,
            base_output_dir=output_dir,
            title_metadata=title_metadata
        )
        total_crops += crops
        
        if idx % 10 == 0 or idx == total_images:
            print(f"\n进度: [{idx}/{total_images}] 已裁剪 {total_crops} 个区域")
    
    # ===== 阶段三：写入每个 title 的元数据 JSON 文件 =====
    print(f"\n{'='*60}")
    print("【阶段三】写入各 title 的 img-caption 元数据 JSON...")
    for title_name, records in title_metadata.items():
        title_folder = os.path.join(output_dir, title_name)
        if not os.path.exists(title_folder):
            continue
        metadata_path = os.path.join(title_folder, "img_caption_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                "title": title_name,
                "pairs": records
            }, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 已生成: {metadata_path} (共 {len(records)} 个 img-caption 对)")
    
    # ===== 统计输出 =====
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"处理图片: {total_images} 张")
    print(f"发现 title: {len(all_titles)} 个")
    print(f"裁剪区域: {total_crops} 个")
    print(f"输出目录: {output_dir}")
    
    print(f"\n【归档结果】")
    title_folders = sorted([d for d in os.listdir(output_dir)
                           if os.path.isdir(os.path.join(output_dir, d)) and 'title' in d])
    for folder in title_folders:
        folder_path = os.path.join(output_dir, folder)
        files = os.listdir(folder_path)
        json_exists = "img_caption_metadata.json" in files
        print(f"  📁 {folder}/ : {len(files)} 个文件 {'(含元数据)' if json_exists else ''}")
    
    print(f"{'='*60}")
    return total_crops


def main():
    """
    主函数：配置参数并运行
    """
    # ========================================
    # 配置参数（根据你的实际情况修改）
    # ========================================
    IMAGES_DIR = r"Test\Test_Images\images\train"   # 图片文件夹
    LABELS_DIR = r"Test\Test_Images\labels\train"   # 标注文件夹
    CLASSES_FILE = r"Test\Test_Images\classes.txt"  # 类别文件
    OUTPUT_DIR = r"Test\Images_village_cropped"         # 输出根目录
    
    process_folder(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        output_dir=OUTPUT_DIR,
        classes_file=CLASSES_FILE
    )


if __name__ == "__main__":
    main()