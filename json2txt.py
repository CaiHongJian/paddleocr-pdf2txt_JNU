import json
import os
from tqdm import tqdm

def process_ocr_json(json_file_path, output_dir="output", indent_threshold=20):
    """
    根据水平起始坐标判断段落，处理 JSON 并生成 TXT。
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

        # 4. 提取文件名
        base_name = os.path.splitext(os.path.basename(json_file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")

        # 5. 主逻辑
        result_text = ""
        total_lines = len(texts)

        # --- 配置缩进样式 ---
        # 这里使用 4 个空格，你也可以改成 c使用制表符
        # indent_style = "    " 
        indent_style = "\t"  # 使用制表符

        with tqdm(total=total_lines, desc="正在处理", unit="行", colour="#ffffff", ncols=100) as pbar:
            for i in range(total_lines):
                current_text = texts[i].strip()
                current_left = boxes[i][0]

                # --- 关键逻辑判断 ---
                if current_left >= indent_threshold:
                    # 情况1：起始坐标 >= 20，判定为“新段落开头”
                    # 修改点：换行符 + 缩进样式 + 当前文字
                    result_text += f"\n{indent_style}{current_text}"
                else:
                    # 情况2：起始坐标 < 20，判定为“上一段落的续行”
                    # 保持原样，直接拼接
                    result_text += current_text

                # --- 更新进度条 ---
                pbar.update(1)
                preview = current_text[:10].replace('\n', '')
                pbar.set_postfix_str(f"当前: {preview}...")

        # 去除最开头可能产生的多余空行和空白字符
        result_text = result_text.lstrip('\n').lstrip()

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

    process_ocr_json(input_json, output_folder)

if __name__ == "__main__":
    main()