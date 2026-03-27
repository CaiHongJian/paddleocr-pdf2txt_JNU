import json
import os
from tqdm import tqdm  # 导入 tqdm 库

def process_ocr_json(json_file_path, output_dir="output", indent_threshold=20):
    """
    根据水平起始坐标判断段落，处理 JSON 并生成 TXT。
    
    参数:
        json_file_path: 输入 JSON 文件路径
        output_dir: 输出文件夹路径
        indent_threshold: 判断新段落的起始 X 坐标阈值 (默认 20)
    """
    try:
        # 1. 读取文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取 JSON 文件: {json_file_path}")

        # 2. 检查数据结构
        if 'rec_texts' not in data or 'rec_boxes' not in data:
            print("错误：JSON 文件中缺少 'rec_texts' 或 'rec_boxes' 字段")
            return

        texts = data['rec_texts']
        boxes = data['rec_boxes']

        # 3. 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 4. 提取文件名（去掉 .json 后缀）
        base_name = os.path.splitext(os.path.basename(json_file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")

        # 5. 主逻辑：遍历每一行数据
        result_text = ""
        total_lines = len(texts)

        # 使用 tqdm 包装 iterable，设置描述和颜色
        # colour='green' 可以让进度条变绿，ncols 设置宽度
        with tqdm(total=total_lines, desc="正在处理", unit="行", colour="#ffffff", ncols=100) as pbar:
            for i in range(total_lines):
                current_text = texts[i].strip()
                current_left = boxes[i][0]

                # --- 关键逻辑判断 ---
                if current_left >= indent_threshold:
                    # 情况1：起始坐标 >= 20，判定为“新段落开头”
                    result_text += f"\n{current_text}"
                else:
                    # 情况2：起始坐标 < 20，判定为“上一段落的续行”
                    result_text += current_text

                # --- 更新进度条 ---
                # 更新计数
                pbar.update(1)
                # 可选：在进度条后缀显示当前处理的文字预览（截取前10个字符）
                preview = current_text[:10].replace('\n', '')
                pbar.set_postfix_str(f"当前: {preview}...")

        # 去除最开头可能产生的多余空行
        result_text = result_text.lstrip('\n')

        # 6. 保存文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result_text)

        print(f"\n✅ 处理完成！文件已保存至: {output_path}")

    except Exception as e:
        print(f"处理文件时发生错误: {e}")

def main():
    # --- 配置区 ---
    input_json = "output/test1_res.json" 
    output_folder = "output"

    # 执行处理
    process_ocr_json(input_json, output_folder)

if __name__ == "__main__":
    main()