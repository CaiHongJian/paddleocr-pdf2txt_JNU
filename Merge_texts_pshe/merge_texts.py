import os
import re

def extract_numbers(filename):
    """
    从文件名中提取页码和n值
    文件名格式: page_页码_n.txt
    """
    match = re.match(r'page_(\d+)(?:_(\d+))?', filename)
    if match:
        page_num = int(match.group(1))
        n_value = int(match.group(2)) if match.group(2) else 0
        return page_num, n_value
    return float('inf'), float('inf')  # 如果格式不匹配，则放在最后

def sort_files_by_page_and_n(folder_path):
    """
    按照页码和n值对文件进行排序
    """
    files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            files.append(filename)
    
    # 使用自定义的排序键函数
    sorted_files = sorted(files, key=extract_numbers)
    return sorted_files

def merge_txt_files(input_folder, output_file):
    """
    将输入文件夹中的所有txt文件按页码和n值排序后合并到输出文件
    """
    sorted_files = sort_files_by_page_and_n(input_folder)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in sorted_files:
            input_path = os.path.join(input_folder, filename)
            
            # 读取原文件内容并写入输出文件
            with open(input_path, 'r', encoding='utf-8') as infile:
                content = infile.read()
                outfile.write(content)
                
            # 在文件之间添加分隔符（可选）
            # outfile.write('\n')  # 添加一个空行作为文件间的分隔
            
    print(f"合并完成！输出文件: {output_file}")
    print(f"共处理了 {len(sorted_files)} 个文件")
    print("处理的文件列表:", sorted_files)

if __name__ == "__main__":
    input_folder = r"OCR_IMGs2TXT_chj\ocr_out_texts"
    output_file = "result.txt"
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹 '{input_folder}' 不存在")
    else:
        merge_txt_files(input_folder, output_file)
