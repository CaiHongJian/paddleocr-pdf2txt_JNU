import os
import glob
import time
from pathlib import Path
from tqdm import tqdm  # 进度条库
from paddleocr import PaddleOCR


def process_images_in_folder(input_folder, output_folder=None, y_threshold=25):
    """
    批量处理文件夹内的JPEG图片，带进度条显示
    
    Args:
        input_folder: 输入图片文件夹路径
        output_folder: 输出TXT文件保存路径
        y_threshold: Y坐标差异阈值
    """
    
    # 初始化PaddleOCR
    print("正在初始化PaddleOCR模型...")
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )
    print("✓ 模型初始化完成\n")
    
    # 设置输出文件夹
    if output_folder is None:
        output_folder = input_folder
    else:
        os.makedirs(output_folder, exist_ok=True)
    
    # 查找所有JPEG图片
    image_extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
    
    image_paths = sorted(image_paths)
    
    if not image_paths:
        print(f"在 {input_folder} 中未找到JPEG图片")
        return
    
    print(f"找到 {len(image_paths)} 张图片，开始批量OCR识别...\n")
    
    # 统计信息
    stats = {
        'success': 0,
        'failed': 0,
        'total_blocks': 0
    }
    
    # 创建总体进度条（外层）
    with tqdm(total=len(image_paths), 
              desc="总体进度", 
              position=0, 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
              postfix="当前: 等待开始") as pbar_total:
        
        # 逐张处理图片
        for idx, img_path in enumerate(image_paths, 1):
            img_name = Path(img_path).stem
            pbar_total.set_postfix_str(f"当前: {img_name[:20]}...")  # 显示当前处理的文件名
            
            try:
                success, blocks_count = process_single_image_with_progress(
                    img_path, output_folder, ocr, y_threshold, idx, len(image_paths)
                )
                if success:
                    stats['success'] += 1
                    stats['total_blocks'] += blocks_count
                else:
                    stats['failed'] += 1
            except Exception as e:
                stats['failed'] += 1
                tqdm.write(f"\n✗ [{idx}/{len(image_paths)}] {img_name}: 处理失败 - {str(e)}")
            
            # 更新总体进度
            pbar_total.update(1)
            # 更新后缀显示统计信息
            pbar_total.set_postfix_str(
                f"成功:{stats['success']} 失败:{stats['failed']} 当前:{img_name[:15]}"
            )
    
    # 打印最终统计
    print(f"\n\n{'='*50}")
    print(f"处理完成！")
    print(f"  总计: {len(image_paths)} 张图片")
    print(f"  成功: {stats['success']} 张")
    print(f"  失败: {stats['failed']} 张")
    print(f"  共识别: {stats['total_blocks']} 个文本块")
    print(f"  输出位置: {output_folder}")
    print(f"{'='*50}")


def process_single_image_with_progress(image_path, output_folder, ocr, y_threshold, 
                                       current_idx, total_count):
    """
    处理单张图片，带详细进度步骤
    
    Returns:
        (success: bool, blocks_count: int)
    """
    
    img_name = Path(image_path).stem
    output_path = os.path.join(output_folder, f"{img_name}.txt")
    
    # 创建单文件进度条（内层，嵌套显示）
    steps = ['读取图片', 'OCR识别', '解析结果', '排序段落', '保存文件']
    
    with tqdm(total=len(steps), 
              desc=f"[{current_idx}/{total_count}] {img_name[:15]:<15}", 
              position=1, 
              leave=False,  # 完成后清除此行，避免屏幕混乱
              bar_format='  {desc} |{bar}| {n_fmt}/{total_fmt} {postfix}') as pbar_step:
        
        # Step 1: 读取图片（其实OCR内部会读，这里模拟进度）
        pbar_step.set_postfix_str("读取中...")
        time.sleep(0.05)  # 微小延迟让进度条可见
        pbar_step.update(1)
        
        # Step 2: OCR识别（最耗时的步骤）
        pbar_step.set_postfix_str("📝 OCR识别中...")
        try:
            result = ocr.predict(input=image_path)
            pbar_step.update(1)
        except Exception as e:
            pbar_step.set_postfix_str(f"✗ OCR失败")
            raise e
        
        # Step 3: 解析结果
        pbar_step.set_postfix_str("🔍 解析结果...")
        text_blocks = parse_ocr_result(result)
        pbar_step.update(1)
        
        # Step 4: 排序段落
        pbar_step.set_postfix_str("📊 排序段落...")
        organized_text = organize_by_paragraphs(text_blocks, y_threshold)
        pbar_step.update(1)
        
        # Step 5: 保存文件
        pbar_step.set_postfix_str("💾 保存中...")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(organized_text)
        pbar_step.update(1)
        pbar_step.set_postfix_str(f"✓ 完成({len(text_blocks)}块)")
    
    # 使用tqdm.write输出完成信息（避免与进度条冲突）
    tqdm.write(f"✓ [{current_idx}/{total_count}] {img_name}: 识别 {len(text_blocks)} 个文本块 → {img_name}.txt")
    
    return True, len(text_blocks)


def parse_ocr_result(result):
    """
    解析OCR结果，提取文本块信息
    """
    text_blocks = []
    
    for res in result:
        # 处理不同的返回格式
        if hasattr(res, 'json'):
            try:
                data = res.json()
                for item in data.get('ocr_result', []):
                    if isinstance(item, dict):
                        box = item.get('box', [])
                        text = item.get('text', '')
                        score = item.get('score', 0)
                        if text and box:
                            center_y = sum(point[1] for point in box) / len(box)
                            min_x = min(point[0] for point in box)
                            text_blocks.append({
                                'text': text,
                                'confidence': score,
                                'center_y': center_y,
                                'min_x': min_x,
                                'box': box
                            })
            except:
                pass
        
        # 备用解析
        elif isinstance(res, list):
            for line in res:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    box = line[0] if isinstance(line[0], list) else []
                    text_info = line[1] if len(line) > 1 else ('', 0)
                    
                    if isinstance(text_info, (list, tuple)):
                        text = text_info[0]
                        score = text_info[1] if len(text_info) > 1 else 0
                    else:
                        text = str(text_info)
                        score = 0
                    
                    if text and box and isinstance(box, list) and len(box) >= 4:
                        try:
                            center_y = sum(point[1] for point in box) / len(box)
                            min_x = min(point[0] for point in box)
                            text_blocks.append({
                                'text': text,
                                'confidence': score,
                                'center_y': center_y,
                                'min_x': min_x,
                                'box': box
                            })
                        except:
                            pass
    
    return text_blocks


def organize_by_paragraphs(text_blocks, y_threshold=30):
    """
    将文本块按段落结构组织
    """
    if not text_blocks:
        return ""
    
    # 按Y坐标排序（从上到下）
    text_blocks.sort(key=lambda x: x['center_y'])
    
    # 按行分组
    lines = []
    current_line = [text_blocks[0]]
    current_y = text_blocks[0]['center_y']
    
    for block in text_blocks[1:]:
        if abs(block['center_y'] - current_y) < y_threshold:
            current_line.append(block)
        else:
            current_line.sort(key=lambda x: x['min_x'])
            lines.append(current_line)
            current_line = [block]
            current_y = block['center_y']
    
    if current_line:
        current_line.sort(key=lambda x: x['min_x'])
        lines.append(current_line)
    
    # 合并文本
    paragraphs = []
    for line in lines:
        line_text = ' '.join([block['text'] for block in line])
        paragraphs.append(line_text)
    
    return '\n'.join(paragraphs)


def main():
    # ==================== 配置区域 ====================
    
    # 输入文件夹路径
    INPUT_FOLDER = r"BookImages"
    
    # 输出文件夹路径（None表示与图片同文件夹）
    OUTPUT_FOLDER = r"ocr_out_texts"
    
    # Y坐标阈值（判断同一行的像素差）
    Y_THRESHOLD = 25
    
    # =================================================
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"错误: 输入文件夹不存在: {INPUT_FOLDER}")
        return
    
    # 检查tqdm是否安装
    try:
        import tqdm
    except ImportError:
        print("正在安装 tqdm 进度条库...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'tqdm'])
        print("✓ 安装完成，请重新运行程序")
        return
    
    process_images_in_folder(INPUT_FOLDER, OUTPUT_FOLDER, Y_THRESHOLD)


if __name__ == "__main__":
    main()