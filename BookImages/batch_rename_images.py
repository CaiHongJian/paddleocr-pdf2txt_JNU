import os
import shutil
import re

def batch_rename_images(source_folder, target_folder):
    """
    批量重命名图片：
    输入格式：page_0023_L_1.jpg (页码_左右_序号)
    输出格式：page_0023_1.jpg   (页码_全局序号)
    """
    
    # 1. 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"❌ 错误：找不到源文件夹 {source_folder}")
        return

    # 2. 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"📁 已创建目标文件夹: {target_folder}")
    else:
        # 如果文件夹存在且不为空，建议清空或提示（这里不做清空处理，直接覆盖同名文件）
        print(f"📁 目标文件夹已存在: {target_folder}")

    # 3. 获取所有图片文件
    all_files = os.listdir(source_folder)
    # 过滤出 jpg 或 png 文件
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG'))]
    
    if not image_files:
        print("⚠️ 警告：源文件夹中没有找到图片文件。")
        return

    print(f"🔍 发现 {len(image_files)} 个图片文件，开始处理...\n")

    # 4. 按页码分组
    # 使用字典存储，key是页码（如 '0023'），value是该页码下的所有文件列表
    pages_dict = {}
    
    for filename in image_files:
        # 解析文件名，假设格式严格为 page_XXXX_L_1.jpg
        # 使用正则提取页码
        match = re.match(r'page_(\d+)_', filename)
        if match:
            page_num = match.group(1)
            if page_num not in pages_dict:
                pages_dict[page_num] = []
            pages_dict[page_num].append(filename)
        else:
            print(f"⚠️ 跳过无法识别格式的文件: {filename}")

    # 5. 处理每一页
    total_processed = 0
    
    # 对页码进行排序（保证按 0001, 0002... 顺序处理）
    sorted_pages = sorted(pages_dict.keys())

    for page_num in sorted_pages:
        files_in_page = pages_dict[page_num]
        
        # --- 核心排序逻辑 ---
        # 排序规则：先按 L/R 排序（L 在前，R 在后），再按最后的数字排序
        # 比如：L_1, R_1, L_2, R_2
        def sort_key(filename):
            # 提取 L/R 和 最后的数字
            # 匹配 page_0023_L_1.jpg -> 提取 ('L', 1)
            match = re.search(r'_(L|R)_(\d+)\.', filename)
            if match:
                side = match.group(1) # 'L' or 'R'
                order = int(match.group(2)) # 1, 2...
                # 排序优先级：先按 order 排（1, 2），再按 side 排（L, R）
                # 这样顺序就是：L_1 -> R_1 -> L_2 -> R_2
                return (order, side)
            return (999, 'Z') # 兜底

        # 对当前页的文件进行排序
        files_in_page.sort(key=sort_key)

        # 开始重命名并复制
        for index, old_filename in enumerate(files_in_page):
            # 新的序号 (从 1 开始)
            new_sequence = index + 1
            
            # 构建新文件名：page_页码_新序号.jpg
            # 保持原文件的后缀名
            file_ext = os.path.splitext(old_filename)[1]
            new_filename = f"page_{page_num}_{new_sequence}{file_ext}"
            
            # 构建完整路径
            src_path = os.path.join(source_folder, old_filename)
            dst_path = os.path.join(target_folder, new_filename)
            
            # 复制并重命名
            try:
                shutil.copy2(src_path, dst_path) # copy2 可以保留修改时间等元数据
                # print(f"复制: {old_filename} -> {new_filename}")
            except Exception as e:
                print(f"❌ 复制失败 {old_filename}: {e}")
            
            total_processed += 1

    print(f"✅ 处理完成！共处理 {total_processed} 个文件。")
    print(f"📂 结果已保存至: {target_folder}")

# --- 主程序入口 ---
if __name__ == "__main__":
    # 请在这里修改你的文件夹路径
    SOURCE_DIR = r"BookImages\All_Images"       # 原图片文件夹路径
    TARGET_DIR = r"BookImages\Renamed_All_Images"  # 重命名后保存的文件夹路径

    batch_rename_images(SOURCE_DIR, TARGET_DIR)