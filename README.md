# paddleocr-pdf2txt_JNU

基于PaddleOCR与YOLO的双栏PDF正文提取系统。专门用于处理《全粤村情》PDF文档，实现村落名称识别、正文提取、插图分离等全流程自动化。隶属于张子邦光电创新实验课程小组。

## 项目功能概述

本项目实现了从PDF文档中自动化提取各村村落信息的完整流水线：

1. **PDF转图片**：将PDF文档批量转换为高清图片
2. **YOLO目标检测**：使用YOLO模型识别页面中的不同元素（标题、正文、插图、图注等）
3. **智能分割归档**：根据检测结果裁剪图片，并按村落标题归属自动分类归档
4. **OCR文字识别**：使用PaddleOCR对裁剪后的图片进行文字识别
5. **智能文本合并**：根据缩进判断段落结构，自动合并同一村落的完整文本
6. **插图重命名**：通过识别图注文字为插图自动命名

---

## 整体文件结构

```
paddleocr-pdf2txt_JNU/
├── data/                          # 数据目录
│   ├── Dataset_village/           # 村落数据集
│   │   ├── classes.txt            # 类别定义文件
│   │   ├── village_dataset.yaml   # YOLO数据集配置
│   │   └── 数据集文件结构.md
│   ├── Final_output/             # 最终输出结果
│   │   └── 广州市从化区卷一_1-260.txt
│   └── Temp_data/                # 临时数据目录
│       └── images_cropped_villages/  # 按村落标题分类裁剪的中间结果
│           └── Page_022_title/   # 单个村落的裁剪目录
│               ├── img_caption_metadata.json  # 插图-图注配对元数据
│               └── 各类裁剪图片
├── docs/                          # 文档目录
│   ├── 汇报记录/                  # 项目汇报文档
│   ├── crop_by_yolo_with_metadata输出格式.md
│   └── 各村OCR结果文件结构.md
├── models/                        # 模型文件目录
│   ├── Model_xhao/                # 布局检测模型
│   │   ├── detect_layout.pt
│   │   └── detect_text.pt
│   ├── Village_Model_chj/         # 村落专用检测模型_chj
│   │   ├── Village_V0.pt
│   │   ├── Village_V1.pt
│   │   ├── classes.txt
│   │   └── village_dataset.yaml
│   └── yolo11n.pt                 # YOLO11n预训练模型
├── pipelines/                     # 流水线脚本目录
│   ├── Step1_YOLO_detect/         # 步骤1：YOLO检测
│   │   ├── predict_images.py      # 图片预测脚本
│   │   └── train_model_chj.py     # YOLO模型训练脚本
│   └── Step2_Crop_by_YOLO_Label/  # 步骤2：按标注裁剪
│       └── crop_by_yolo_with_metadata.py  # 增强版分割归档脚本
├── util/                          # 工具模块目录
│   ├── detect_pdf_yolo_xhao.py    # PDF级别的YOLO检测
│   ├── ocr_utils.py               # PaddleOCR工具封装
│   ├── pdf_to_images.py           # PDF转图片工具
│   ├── txt_extractor.py           # 带缩进判断的文本提取
│   └── txt_merger.py              # 多片段文本合并
├── .gitignore                     # Git忽略文件配置
├── process_cropped_data.py       # 主处理脚本：处理裁剪数据并生成最终结果
└── README.md                      # 项目说明文档
```

---

## 流水线详细说明

### 步骤1：PDF转图片
**文件**：`util/pdf_to_images.py`

功能：
- 交互式输入PDF路径和输出目录
- 使用PyMuPDF将PDF每页转换为300DPI高清PNG图片
- 自动按 `Page_001.png`、`Page_002.png` 格式命名

### 步骤2：YOLO目标检测
**文件**：`pipelines/Step1_YOLO_detect/predict_images.py`

功能：
- 加载训练好的YOLO模型
- 批量检测所有PDF页面图片
- 输出YOLO格式的标注文件（.txt）

检测类别通常包含：
- `title`：村落标题
- `txt`：正文文本
- `img`：插图
- `caption`：插图附文

### 步骤3：智能分割归档
**文件**：`pipelines/Step2_Crop_by_YOLO_Label/crop_by_yolo_with_metadata.py`

核心特性：
1. **两次扫描策略**：
   - 第一遍扫描：收集所有页面的title坐标信息，建立全局title索引
   - 第二遍扫描：根据归属关系将每个元素分配到最近的title下

2. **跨页归属逻辑**：
   - 某个title的作用域从其出现位置开始，直到下一个title出现前（跨页有效）

3. **插图-图注配对**：
   - 规则：每个caption匹配其上方最近的img（y_center在caption下方且距离最小）
   - 生成 `img_caption_metadata.json` 记录配对信息

输出目录结构示例：
```
images_cropped_villages/
├── Page_022_title/
│   ├── Page_022_title.png
│   ├── Page_022_txt_1.png
│   ├── Page_022_img.png
│   ├── Page_022_caption.png
│   └── img_caption_metadata.json
├── Page_022_title/
└── ...
```

### 步骤4：OCR识别与最终结果生成
**文件**：`process_cropped_data.py`

处理流程：
1. **识别村落名称**：对每个title图片进行OCR，得到真实村名
2. **识别图注并重命名插图**：利用元数据配对关系，识别caption文字作为插图文件名
3. **带缩进的文本提取**：
   - 模块：`util/txt_extractor.py`
   - 根据文本框的X坐标判断是否为缩进段落
   - 缩进达到阈值的行前面添加制表符`\t`
4. **智能文本合并**：
   - 模块：`util/txt_merger.py`
   - 按页码和txt编号排序所有文本片段
   - 根据片段首行是否有制表符`\t`决定是否换行

最终输出目录结构：
```
各村OCR结果/
├── 1_大围村/
│   ├── 1_大围村.txt
│   ├── 大围村村貌（摄于2017年，从化区档案局提供）.png
│   └── ...
├── 2_沙岗村/
│   └── ...
└── ...
```

---

## 环境配置

### 基础环境要求
- **操作系统**：Windows 10/11
- **Python版本**：Python 3.8 ~ 3.11（推荐3.10）
- **内存**：建议 8GB+
- **GPU（可选）**：NVIDIA显卡，支持CUDA加速

### 依赖库安装

#### 1. PaddleOCR环境
```bash
# 利用Anaconda，以下为经过验证的CPU版本安装命令
conda create -n paddleocr python=3.9 -y
conda activate paddleocr
python -m pip install --upgrade pip
python -m pip install paddleocr
pip install paddlepaddle==3.2.2 
# GPU版本PaddlePaddle：pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
# 注意：一般情况下，PaddlePaddle需要先安装，再安装PaddleOCR
```

#### 2. YOLO环境
```bash
# 以下为AI所给命令，暂未验证
# 基础图像处理库
pip install opencv-python opencv-contrib-python
pip install PyMuPDF

# 深度学习框架 - PyTorch（根据你的CUDA版本选择）
# CPU版本（推荐新手使用）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 或GPU版本（CUDA 11.8）
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# YOLO11（Ultralytics）
pip install ultralytics
```

#### 3. 完整依赖清单（暂未验证）
你也可以创建 `requirements.txt` 文件：
```txt
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
PyMuPDF>=1.23.0
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.200
paddlepaddle>=2.5.0
paddleocr>=2.7.0
numpy>=1.24.0
```

然后一次性安装：
```bash
pip install -r requirements.txt
```

### 验证安装
运行以下代码验证所有依赖是否正确安装：
```python
import cv2
import fitz
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR

print("✅ 所有依赖库导入成功！")
print(f"OpenCV版本: {cv2.__version__}")
print(f"PyMuPDF版本: {fitz.__version__}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
```

---

## 快速使用指南

### 完整处理流程
1. **准备PDF文件**：将待处理的PDF文档放入指定目录
2. **PDF转图片**：运行 `util/pdf_to_images.py` 转换为高清图片
3. **YOLO检测**：运行 `pipelines/Step1_YOLO_detect/predict_images.py` 生成标注
4. **分割归档**：运行 `pipelines/Step2_Crop_by_YOLO_Label/crop_by_yolo_with_metadata.py`
5. **OCR生成结果**：运行 `process_cropped_data.py` 得到最终各村结果

---

## 技术栈说明

| 组件 | 技术选型 | 用途 |
|------|---------|------|
| PDF处理 | PyMuPDF (fitz) | PDF转高清图片 |
| 目标检测 | YOLO11 | 检测标题、正文、插图、图注 |
| OCR文字识别 | PaddleOCR (PP-OCRv5) | 高精度中文识别 |
| 图像处理 | OpenCV | 图片裁剪、尺寸处理 |
