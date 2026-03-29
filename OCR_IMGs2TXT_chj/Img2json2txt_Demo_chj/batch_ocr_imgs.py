# -*- coding: utf-8 -*-
"""
批量 OCR 处理图片脚本（demo）
功能：读取 BookImages 文件夹图片 -> 识别 -> 将 JSON 结果统一保存到 ocr_out_jsons 文件夹
作者：蔡鸿键
"""

# 1. 导入必要的库
from paddleocr import PaddleOCR
import os
from tqdm import tqdm  # 用于显示进度条的库

# --- 1. 初始化 OCR 引擎 ---
# 严格参照 PP-OCRv5 的参数配置
ocr = PaddleOCR(
    use_doc_orientation_classify=False,  # 不使用文档方向分类
    use_doc_unwarping=False,              # 不使用文档解缠绕
    use_textline_orientation=False         # 不使用文本行方向
    # device="gpu",                       # 通过 device 参数使得在模型推理时使用 GPU
)

# --- 2. 设置文件夹路径 ---
# 2.1 输入图片所在的文件夹（请确保该文件夹在项目根目录下）
image_folder = r"BookImages"

# 2.2 输出 JSON 文件的文件夹（修改为 ocr_out_jsons）
json_output_folder = r"ocr_out_jsons" 
# 注意：这里使用相对路径，表示项目根目录下的 ocr_out_jsons 文件夹

# --- 3. 关键修正：确保输出文件夹存在 ---
# 如果 ocr_out_jsons 文件夹不存在，则自动创建它
# 否则程序会因为找不到路径而报错
if not os.path.exists(json_output_folder):
    os.makedirs(json_output_folder)
    print(f"已自动创建 JSON 输出文件夹: {json_output_folder}")

# --- 4. 获取文件夹内所有图片 ---
# 支持常见的图片格式，将后缀名转为小写进行匹配
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(image_extensions)]

# --- 5. 批量处理（带进度条） ---
print(f"开始处理 {len(image_files)} 张图片...\n")

# 使用 tqdm 包装循环，自动显示进度条
for filename in tqdm(image_files, desc="OCR Processing", unit="image"):
    try:
        # 1. 拼接完整的图片路径（告诉程序图片在哪里读）
        image_path = os.path.join(image_folder, filename)
        
        # 2. 运行 OCR 识别
        # predict 方法返回的是一个包含结果的列表
        result = ocr.predict(image_path)
        
        # 3. 处理结果并保存
        # 遍历识别结果（通常一张图对应一个结果）
        for res in result:
            # --- 修改点：保存 JSON 文件 ---
            # 1. 获取文件名（不含后缀）用于命名 JSON
            # 例如：001.png -> 变成 001.json
            name_without_ext = os.path.splitext(filename)[0] 
            json_filename = f"{name_without_ext}.json"
            
            # 2. 拼接保存路径
            # os.path.join 用来拼接文件夹和文件名
            # 这里将路径改为了 json_output_folder（即 ocr_out_json）
            json_path = os.path.join(json_output_folder, json_filename) 
            
            # 3. 执行保存
            res.save_to_json(json_path)
            
            # --- 保存可视化图片（可选） ---
            # 如果你也想保存画了框的图片，取消下面两行的注释
            # img_filename = f"{name_without_ext}_result.jpg"
            # res.save_to_img(os.path.join(json_output_folder, img_filename))

    except Exception as e:
        # 如果某张图片出错，打印错误但不停止整个程序
        # tqdm.write 用于在进度条上方输出错误信息
        tqdm.write(f"处理文件 {filename} 时出错: {str(e)}")

print("\n 所有图片处理完成！JSON文件已统一保存在 ocr_out_jsons 文件夹中。")