import os
import shutil

def copy_images_from_subdirs(source_root_dir, target_dir_name):
    """
    功能：遍历 source_root_dir 下的所有子文件夹，将其中的 .jpg 图片复制到新的文件夹中。
    
    参数:
    source_root_dir (str): 源文件夹路径 (例如: crops)
    target_dir_name (str): 目标文件夹名称 (例如: allimages)
    """
    
    # 1. 检查源文件夹是否存在
    if not os.path.exists(source_root_dir):
        print(f"❌ 错误：找不到源文件夹 '{source_root_dir}'。请检查路径是否正确。")
        return

    # 2. 确定目标文件夹的完整路径
    # os.path.join 用于智能拼接路径，避免斜杠方向错误
    target_path = os.path.join(os.path.dirname(source_root_dir), target_dir_name)

    # 3. 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print(f"✅ 已创建目标文件夹: {target_path}")
    else:
        print(f"ℹ️  目标文件夹已存在: {target_path}")

    # 计数器，用于统计复制了多少张图片
    count = 0

    # 4. 开始遍历
    # os.walk 会像走路一样遍历目录树
    # root: 当前遍历到的文件夹路径
    # dirs: 当前文件夹下的子文件夹列表 (这里用不到)
    # files: 当前文件夹下的文件列表
    print("🚀 开始扫描和复制图片...")
    
    for root, dirs, files in os.walk(source_root_dir):
        for file in files:
            # 检查文件是否以 .jpg 或 .JPG 结尾 (忽略大小写)
            if file.lower().endswith('.jpg'):
                
                # 获取源文件的完整路径 (例如: crops\page_0022\page_0022_L_1.jpg)
                source_file_path = os.path.join(root, file)
                
                # 获取目标文件的完整路径 (例如: allimages\page_0022_L_1.jpg)
                # 注意：这里直接把文件名拼接到目标大文件夹下，去掉了中间的子文件夹层级
                target_file_path = os.path.join(target_path, file)
                
                # 检查目标位置是否已经有同名文件，防止覆盖（可选逻辑）
                if os.path.exists(target_file_path):
                    print(f"⚠️  跳过 (文件已存在): {file}")
                    continue

                # 执行复制操作
                # shutil.copy2 可以保留原始文件的修改时间等信息
                shutil.copy2(source_file_path, target_file_path)
                
                count += 1
                # 打印进度（每复制10张打印一次，避免刷屏，或者你可以去掉if直接打印）
                # print(f"复制: {file}") 

    print("-" * 30)
    print(f"🎉 完成！总共成功复制了 {count} 张图片。")
    print(f"📂 图片已保存至: {target_path}")

# ==========================================
# 主程序入口 (小白只需关注这里的设置)
# ==========================================
if __name__ == "__main__":
    # 设置：源文件夹名字 (确保这个文件夹就在脚本旁边，或者填入绝对路径)
    SOURCE_FOLDER = r"BookImages\crops"
    
    # 设置：你想生成的目标文件夹名字
    DEST_FOLDER = r"BookImages\allimages"
    
    # 调用函数
    copy_images_from_subdirs(SOURCE_FOLDER, DEST_FOLDER)