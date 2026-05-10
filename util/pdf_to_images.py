import os
import fitz  # PyMuPDF库

def convert_pdf_to_images():
    print("--- PDF转图片工具 ---")
    
    # 1. 获取用户输入的PDF路径
    pdf_path = input("请输入PDF文件的完整路径 (例如 D:\\docs\\test.pdf): ").strip()
    
    # 去除可能存在的首尾引号 (防止复制粘贴时带入)
    if pdf_path.startswith('"') and pdf_path.endswith('"'):
        pdf_path = pdf_path[1:-1]

    # 检查文件是否存在
    if not os.path.exists(pdf_path):
        print(f"❌ 错误：找不到文件 '{pdf_path}'")
        return

    # 2. 获取用户输入的输出文件夹路径
    output_dir = input("请输入图片保存的文件夹路径 (例如 D:\\images\\output): ").strip()
    
    if output_dir.startswith('"') and output_dir.endswith('"'):
        output_dir = output_dir[1:-1]

    # 如果文件夹不存在，自动创建
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"📂 已创建输出目录: {output_dir}")
        except Exception as e:
            print(f"❌ 无法创建目录: {e}")
            return

    # 3. 设置转换参数
    dpi = 300  # 分辨率
    img_format = "png" # 图片格式
    
    print(f"\n🚀 正在处理: {os.path.basename(pdf_path)} ...")

    try:
        # 打开PDF
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        
        # 计算缩放比例 (72是PDF标准DPI)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        for page_num in range(page_count):
            page = doc.load_page(page_num)
            
            # 渲染页面
            pix = page.get_pixmap(matrix=mat)
            
            # --- 核心修改：自定义文件名格式 ---
            # :03d 表示将数字格式化为至少3位，不足的前面补0 (例如 1 -> 001)
            filename = f"Page_{page_num + 1:03d}.{img_format}"
            output_path = os.path.join(output_dir, filename)
            
            # 保存图片
            pix.save(output_path)
            print(f"  ✅ 已保存: {filename}")
            
        doc.close()
        print(f"\n✨ 转换成功！共 {page_count} 页，保存在: {output_dir}")

    except Exception as e:
        print(f"\n❌ 发生未知错误: {e}")

if __name__ == "__main__":
    convert_pdf_to_images()
    # 保持窗口打开（如果在命令行直接双击运行）
    input("\n按回车键退出...")