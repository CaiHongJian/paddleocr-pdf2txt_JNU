"""
模型训练主程序
    - 功能：基于YOLO框架自动训练目标检测模型
    
    - 使用说明：
        1. 可修改参数
        2. 修改后直接在VSCode中运行此文件（右键 -> Run Python File）
        3. 无需命令行参数，所有设置都在代码中
        4. 训练过程中按Ctrl+C可安全中断
        
    - 作者：chj
    - 日期：2026.2.4
"""


from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO(r"Models_chj\XiongChuMo\XiongChuMoV2.1.pt")  # 加载预训练权重
    
    model.train(
        # 数据集配置
        data=r"Datasets_chj\XiongChuMo\XiongChuMo.yaml",  
        # YAML文件包含：训练/验证图像路径、类别名称、类别数量等元数据
        
        # 训练参数
        epochs=20,            # 训练轮数：完整遍历数据集50次（建议值：30-100）
        imgsz=640,            # 输入图像分辨率：640x640像素（YOLO标准尺寸，可选320/1280）
        batch=16,             # 自动批次大小：CPU运行固定为16
        cache=False,          # 禁用磁盘缓存：节省内存但降低数据加载速度（内存充足可设为True）
        workers=0,            # 数据加载线程数：Windows系统必须设为0，避免多进程崩溃（Linux可设为4-8）
        
        # 输出配置
        project="TrainModels_chj\TrainResults",  # 项目根目录：所有训练结果保存于此文件夹
        name="XiongChuMoV3",    # 实验名称：本次训练结果保存在 TrainResults/[name]/ 下
    )
    
    # 训练完成后输出文件说明：
    # - best.pt / last.pt：最佳/最新模型权重（位于 TrainResults/XiongChuMo/weights/）
    # - results.png：训练指标可视化曲线（损失、mAP、精确率等）
    # - confusion_matrix.png：混淆矩阵（分类误差分析）
    # - args.yaml：本次训练使用的完整参数记录
    # - labels.jpg：数据集标签分布可视化