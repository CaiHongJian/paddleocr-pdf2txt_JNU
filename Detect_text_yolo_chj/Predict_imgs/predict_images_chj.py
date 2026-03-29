"""
    2026.2.4, chj。自动设置保存结果的子文件夹名称为输入文件名
    参数选项设置可参考官方：https://docs.ultralytics.com/modes/predict/#inference-arguments
"""
import os
from ultralytics import YOLO

# 设置图片路径
imgs_path = r"test.jpg"  # 替换为你的图片路径

# 从路径中提取文件名（不含扩展名）
file_name = os.path.splitext(os.path.basename(imgs_path))[0]

model = YOLO(r"Models_chj\XiongChuMo\best.pt") # 使用的预测模型
"""
    参数选项设置可参考官方：https://docs.ultralytics.com/modes/predict/#inference-arguments 
"""
model.predict(
    source=imgs_path,   # 待预测目标
    save=True,
    show=False,
    # save_txt=True,       # 保存检测结果yolo格式标签文件
    project="Predict_chj\PredictResults",  # 预测结果保存目录
    name=file_name   # 使用输入文件名作为结果文件夹名称

)