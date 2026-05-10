"""
模型训练主程序
    - 功能：基于YOLO框架自动训练目标检测模型
"""
from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO(r"models\Village_V1.pt")  # 加载预训练权重
    
    model.train(
        # 数据集配置
        data=r"data\Dataset_village\village_dataset.yaml",  
        # YAML文件包含：训练/验证图像路径、类别名称、类别数量等元数据
        
        # 训练参数
        epochs=30,            # 训练轮数：完整遍历数据集50次（建议值：30-100）
        imgsz=640,            # 输入图像分辨率：640x640像素（YOLO标准尺寸，可选320/1280）
        batch=16,             # 自动批次大小：CPU运行固定为16
        cache=False,          # 禁用磁盘缓存：节省内存但降低数据加载速度（内存充足可设为True）
        workers=0,            # 数据加载线程数：Windows系统必须设为0，避免多进程崩溃（Linux可设为4-8）
        # device=0,             # 【新增】指定设备：0代表第一块GPU (cuda:0)。若想用CPU则填 'cpu'
        
        # 输出配置
        project="Models_Villages",  # 项目根目录：所有训练结果保存于此文件夹
        name="VillageModel_V2",    # 实验名称：本次训练结果保存在 TrainResults/[name]/ 下
    )
    
    # 训练完成后输出文件说明：
    # - best.pt / last.pt：最佳/最新模型权重
    # - results.png：训练指标可视化曲线（损失、mAP、精确率等）
    # - confusion_matrix.png：混淆矩阵（分类误差分析）
    # - args.yaml：本次训练使用的完整参数记录
    # - labels.jpg：数据集标签分布可视化