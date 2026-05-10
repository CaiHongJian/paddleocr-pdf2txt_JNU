"""
YOLO 批量检测脚本（LabelImg 兼容版）
功能：
1. 检测图片文件夹内所有图片
2. 自动保存带检测框的可视化结果
3. 保存标准 YOLO 格式 txt 标注文件（5个值，兼容 LabelImg）
"""

import os
import glob
from pathlib import Path
from ultralytics import YOLO


def detect_images(
    model_path: str,
    input_dir: str,
    output_dir: str,
    conf_threshold: float = 0.2,
    imgsz: int = 1024,
    device: str = "cpu",
    save_vis: bool = True,
    save_txt: bool = True,
    line_width: int = 3
):
    """
    批量检测图片并保存结果
    """
    # ========== 1. 创建输出目录 ==========
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    txt_dir = os.path.join(output_dir, "labels")
    
    if save_txt:
        Path(txt_dir).mkdir(parents=True, exist_ok=True)
    
    # ========== 2. 加载模型 ==========
    print(f"正在加载模型: {model_path}")
    model = YOLO(model_path)
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
        img_name = Path(img_path).stem
        img_ext = Path(img_path).suffix
        
        print(f"[{idx}/{total}] 正在处理: {img_name}{img_ext}")
        
        # ========== 使用 ultralytics 内置保存功能 ==========
        det_res = model.predict(
            img_path,
            imgsz=imgsz,
            conf=conf_threshold,
            device=device,
            verbose=False,
            save=save_vis,
            project=output_dir,
            name="visualized",
            exist_ok=True,
            line_width=line_width
        )
        
        result = det_res[0]
        
        # ========== 5. 保存标准 YOLO 格式 txt（5个值，兼容 LabelImg）==========
        if save_txt:
            txt_path = os.path.join(txt_dir, f"{img_name}.txt")
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                for box in result.boxes:
                    class_id = int(box.cls)
                    xywhn = box.xywhn.tolist()[0]
                    x_center, y_center, width, height = xywhn
                    
                    # 标准 YOLO 格式：class x_center y_center width height
                    # 只有 5 个值，去掉 confidence，兼容 LabelImg
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            print(f"      ✓ 标注文件已保存: {txt_path} ({len(result.boxes)} 个目标)")
        
        # ========== 6. 打印检测详情 ==========
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
        vis_dir = os.path.join(output_dir, "visualized")
        print(f"可视化结果保存位置: {vis_dir}")
    if save_txt:
        print(f"标注文件保存位置: {txt_dir}")


def main():
    """
    主函数：配置参数并运行检测
    """
    # ========== 配置参数（根据你的实际情况修改）==========
    
    # 模型配置
    MODEL_PATH = r"models\Village_V1.pt"
    
    # 路径配置
    INPUT_DIR = r"Imgs_testModel"   # 输入图片文件夹
    OUTPUT_DIR = r"pipelines\Step1_YOLO_detect\PredictResults"  # 输出结果文件夹
    
    # 检测参数
    CONF_THRESHOLD = 0.2
    IMGSZ = 1024
    DEVICE = "cpu"
    
    # 可视化参数
    SAVE_VIS = True
    SAVE_TXT = True
    LINE_WIDTH = 3
    
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
        line_width=LINE_WIDTH
    )


if __name__ == "__main__":
    main()