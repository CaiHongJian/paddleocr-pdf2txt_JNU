import json
import os
import sys

# ==========================================
# 模块 1: 核心转换逻辑
# 功能：负责把单个 JSON 内容读取并转换成 YOLO 格式字符串
# ==========================================
def convert_single_json_content(json_data, output_txt_path):
    """
    解析 JSON 数据并写入 TXT 文件
    """
    try:
        # 1. 获取图片尺寸
        # 如果 JSON 里没有这两个字段，程序会报错，提示你检查文件
        img_width = json_data['imageWidth']
        img_height = json_data['imageHeight']
        
        # 2. 打开文件准备写入
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            shapes = json_data.get('shapes', [])
            
            for shape in shapes:
                # 只处理矩形框 (rectangle)
                if shape.get('shape_type') == 'rectangle' and len(shape['points']) == 2:
                    
                    # --- 坐标计算 ---
                    # 获取左上角 (x1, y1) 和 右下角 (x2, y2)
                    x1, y1 = shape['points'][0]
                    x2, y2 = shape['points'][1]
                    
                    # 计算中心点 (x, y) 和 宽高 (w, h)
                    x_center = (x1 + x2) / 2.0
                    y_center = (y1 + y2) / 2.0
                    width = x2 - x1
                    height = y2 - y1
                    
                    # 归一化 (除以图片宽高，变成 0-1 之间的小数)
                    x_center /= img_width
                    y_center /= img_height
                    width /= img_width
                    height /= img_height
                    
                    # --- 类别 ID 处理 ---
                    # 假设你的 label 是 "1", "2"，这里自动转为 0, 1 (YOLO 习惯从 0 开始)
                    # 如果你的 label 是文字（如 'cat'），这里需要改写成字典映射
                    try:
                        class_id = int(shape['label']) - 1 
                    except ValueError:
                        print(f"    ⚠️ 警告: 无法识别的标签 '{shape['label']}'，已跳过。")
                        continue

                    # --- 写入一行数据 ---
                    # 格式: class_id x_center y_center width height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        return True # 返回成功标记

    except KeyError as e:
        print(f"    ❌ 错误: JSON 格式缺失字段 {e}")
        return False
    except Exception as e:
        print(f"    ❌ 未知错误: {e}")
        return False

# ==========================================
# 模块 2: 批量处理控制器
# 功能：扫描文件夹，循环调用模块 1
# ==========================================
def batch_convert(input_folder, output_folder):
    """
    批量处理主函数
    input_folder: 存放 JSON 文件的文件夹路径
    output_folder: 存放 TXT 文件的文件夹路径
    """
    # 1. 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"❌ 错误: 找不到输入文件夹 -> {input_folder}")
        return

    # 2. 自动创建输出文件夹 (如果不存在)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 已自动创建输出文件夹: {output_folder}")

    # 3. 获取所有 JSON 文件列表
    json_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.json')]
    
    if not json_files:
        print(f"⚠️ 警告: 在 {input_folder} 中没有找到任何 .json 文件！")
        return

    print(f"🚀 开始批量处理，共发现 {len(json_files)} 个文件...")

    # 4. 循环处理
    success_count = 0
    fail_count = 0

    for i, filename in enumerate(json_files):
        # 构建完整路径
        json_path = os.path.join(input_folder, filename)
        
        # 输出文件名：把 .json 后缀换成 .txt
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(output_folder, txt_filename)

        # 打印进度 (简单的进度条)
        sys.stdout.write(f'\r正在处理: [{i+1}/{len(json_files)}] {filename} ...')
        sys.stdout.flush()

        # 读取 JSON 文件内容
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 调用核心转换函数
            if convert_single_json_content(data, txt_path):
                success_count += 1
            else:
                fail_count += 1
                
        except Exception as e:
            print(f"\n    ❌ 读取文件失败 {filename}: {e}")
            fail_count += 1

    # 5. 处理完成报告
    print("\n" + "="*40)
    print("✅ 批量处理完成！")
    # print(f"📊 成功: {success_count} 个")
    # print(f"📊 失败: {fail_count} 个")
    print(f"📂 结果保存在: {output_folder}")
    print("="*40)

# ==========================================
#                主程序入口
# ==========================================
if __name__ == "__main__":
    # 要转换的 JSON 文件夹路径
    INPUT_DIR = r"D:\Projects_chj\Detect_text_yolo_chj\newDatasets\images\train" 
    # 转换后 TXT 文件的输出路径
    OUTPUT_DIR = r"D:\Projects_chj\Detect_text_yolo_chj\newDatasets\labels\train"
    
    batch_convert(INPUT_DIR, OUTPUT_DIR)
    