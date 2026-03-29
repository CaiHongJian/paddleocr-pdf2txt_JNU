"""
    2026.2.3，chj。打印使用的模型的任务类型。
"""
from ultralytics import YOLO # type: ignore

model = YOLO(r"yolo11n.pt") # 使用的预测模型，此模型用来目标检测
print(model.task)
print(model.names)