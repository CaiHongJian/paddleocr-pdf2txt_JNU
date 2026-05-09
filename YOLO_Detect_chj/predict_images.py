"""
YOLO 批量检测脚本
功能：
1. 检测图片文件夹内所有图片
2. 保存带检测框的可视化结果（方便人工检查）
3. 保存 YOLO 格式的 txt 标注文件（class x_center y_center width height）
"""

import os
import cv2
import glob
from pathlib import Path
from ultralytics import YOLO


def detect_images(
    model_path: str,                    # 模型路径或 HuggingFace 模型名
    input_dir: str,                     # 输入图片文件夹
    output_dir: str,                    # 输出结果文件夹
    conf_threshold: float = 0.2,        # 置信度阈值
    imgsz: int = 1024,                  # 推理图像尺寸
    device: str = "cpu",                # 设备：cuda:0 或 cpu
    save_vis: bool = True,              # 是否保存可视化结果
    save_txt: bool = True,              # 是否保存 txt 标注文件
    line_width: int = 3,                # 框线宽度
    font_size: int = 15                 # 标签字体大小
):
    """
    批量检测图片并保存结果
    """
    # ========== 1. 创建输出目录 ==========
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    vis_dir = os.path.join(output_dir, "visualized")   # 带框图片保存路径
    txt_dir = os.path.join(output_dir, "labels")       # txt 标注保存路径
    
    if save_vis:
        Path(vis_dir).mkdir(parents=True, exist_ok=True)
    if save_txt:
        Path(txt_dir).mkdir(parents=True, exist_ok=True)
    
    # ========== 2. 加载模型 ==========
    print(f"正在加载模型: {model_path}")
    if os.path.exists(model_path):
        model = YOLO(model_path)
    else:
        # 从 HuggingFace 自动下载
        model = YOLO.from_pretrained(model_path)
    print("模型加载完成！")
    
    # ========== 3. 获取所有图片 ==========
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    image_files = sorted(list(set(image_files)))
    total = len(image_files)
    
    if total == 0:
        print(f"警告: 在 {input_dir} 中未找到图片文件")
        return
    
    print(f"共找到 {total} 张图片，开始检测...")
    print("-" * 60)
    
    # ========== 4. 批量检测 ==========
    for idx, img_path in enumerate(image_files, 1):
        img_name = Path(img_path).stem          # 不含扩展名的文件名
        img_ext = Path(img_path).suffix         # 扩展名
        
        print(f"[{idx}/{total}] 正在处理: {img_name}{img_ext}")
        
        # 执行检测
        det_res = model.predict(
            img_path,
            imgsz=imgsz,
            conf=conf_threshold,
            device=device,
            verbose=False          # 关闭单张图片的详细输出
        )
        
        result = det_res[0]        # 获取检测结果
        
        # 获取图片原始尺寸（用于归一化坐标）
        img_h, img_w = result.orig_shape
        
        # ========== 5. 保存可视化结果（带框图片）==========
        if save_vis:
            annotated_frame = result.plot(
                pil=True,
                line_width=line_width,
                font_size=font_size
            )
            vis_path = os.path.join(vis_dir, f"{img_name}{img_ext}")
            cv2.imwrite(vis_path, annotated_frame)
            print(f"      ✓ 可视化结果已保存: {vis_path}")
        
        # ========== 6. 保存 YOLO 格式 txt 标注文件 ==========
        if save_txt:
            txt_path = os.path.join(txt_dir, f"{img_name}.txt")
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                if len(result.boxes) == 0:
                    # 没有检测到目标，保存空文件
                    pass
                
                for box in result.boxes:
                    class_id = int(box.cls)           # 类别编号
                    confidence = float(box.conf)       # 置信度
                    
                    # 获取归一化坐标 (x_center, y_center, width, height)
                    xywhn = box.xywhn.tolist()[0]     # [x_center, y_center, width, height]
                    
                    x_center, y_center, width, height = xywhn
                    
                    # 写入 txt：class x_center y_center width height confidence
                    # 标准 YOLO 格式不含 confidence，但加上方便后续筛选
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.6f}\n")
            
            print(f"      ✓ 标注文件已保存: {txt_path} (检测到 {len(result.boxes)} 个目标)")
        
        # 打印检测详情
        if len(result.boxes) > 0:
            class_counts = {}
            for box in result.boxes:
                cls_name = result.names[int(box.cls)]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            print(f"      检测详情: {class_counts}")
        else:
            print(f"      ⚠ 未检测到任何目标")
    
    print("-" * 60)
    print(f"检测完成！共处理 {total} 张图片")
    if save_vis:
        print(f"可视化结果保存位置: {vis_dir}")
    if save_txt:
        print(f"标注文件保存位置: {txt_dir}")


def main():
    """
    主函数：配置参数并运行检测
    """
    # ========== 配置参数（根据你的实际情况修改）==========
    
    # 模型配置
    MODEL_PATH = r"Models_Villages\best.pt"             # 或本地模型路径
    
    # 路径配置
    INPUT_DIR = r"Imgs_testModel"              # 输入图片文件夹
    OUTPUT_DIR = r"PredictResults"       # 输出结果文件夹
    
    # 检测参数
    CONF_THRESHOLD = 0.2      # 置信度阈值（越低检出越多，但可能误检）
    IMGSZ = 1024              # 推理尺寸（可选: 640, 1024, 1280, 1600）
    DEVICE = "cpu"            # 设备: "cuda:0" 或 "cpu"
    
    # 可视化参数
    SAVE_VIS = True           # 是否保存带框图片
    SAVE_TXT = True           # 是否保存 txt 标注
    LINE_WIDTH = 3            # 框线宽度
    FONT_SIZE = 15            # 字体大小
    
    # ========== 运行检测 ==========
    detect_images(
        model_path=MODEL_PATH,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        conf_threshold=CONF_THRESHOLD,
        imgsz=IMGSZ,
        device=DEVICE,
        save_vis=SAVE_VIS,
        save_txt=SAVE_TXT,
        line_width=LINE_WIDTH,
        font_size=FONT_SIZE
    )


if __name__ == "__main__":
    main()