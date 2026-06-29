# paddleocr-pdf2txt_JNU

基于 PaddleOCR 与 YOLO 的双栏 PDF 正文提取系统，专门用于处理《全粤村情》文档。系统实现从 PDF 解析、目标检测、裁剪归档、OCR 识别到按村落输出的最终自动化流水线。

## 项目功能概述

本项目实现了从 PDF 文档中自动提取各村村落信息的完整流程：

1. **PDF 转图片**：将 PDF 文档批量转换为高清图片
2. **YOLO 目标检测**：识别页面中的标题、正文、插图、图注等元素
3. **智能分割归档**：根据检测结果裁剪图片，并按村落标题归属自动分类
4. **OCR 文字识别**：使用 PaddleOCR 对裁剪图片进行文字识别
5. **智能文本合并**：根据缩进规则判断段落结构，合并同一村落的完整文本
6. **插图重命名**：识别图注文字并作为插图文件名保存

---

## 整体文件结构

```
paddleocr-pdf2txt_JNU/
├── data/
│   ├── Dataset_village/         # 部分村落数据集（YOLO训练用）
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   └── val/
│   │   ├── classes.txt
│   │   ├── village_dataset.yaml
│   │   └── 数据集文件结构.md
│   ├── Final_output/            # 最终输出结果目录
│   │   ├── 各村OCR结果/
│   │   └── 广州市从化区卷一_1-260.txt
│   └── Temp_data/               # 临时数据目录（中间结果）
│       ├── images_PDF/          # 存放PDF转换后的图片及YOLO标注图片
│       ├── images_cropped_villages/   # 按村落标题分类裁剪的结果
│       └── ocr_json_results/
├── docs/                       # 文档目录
│   ├── 汇报记录/
│   ├── crop_by_yolo_with_metadata输出格式.md
│   └── 各村OCR结果文件结构.md
├── models/                            # 模型文件目录
│   ├── Model_xhao/
│   │   ├── detect_layout.pt           # 版面布局检测模型
│   │   └── detect_text.pt             # 文本区域检测模型
│   ├── Village_Model_chj/             # （废弃）
│   └── yolo11n.pt
├── pipelines/
│   ├── Step1_YOLO_detect/
│   │   └── detect_pdf_yolo_xhao.py
│   └── Step2_Crop_by_YOLO_Label/
│       └── crop_by_yolo_with_metadata.py
├── util/
│   ├── ocr_utils.py
│   ├── pdf_to_images.py
│   ├── txt_extractor.py
│   └── txt_merger.py
├── process_cropped_data.py
├── .gitignore
└── README.md
```

### data 目录说明

#### 1. Dataset_village/ - YOLO 训练数据集

**classes.txt 定义的 7 个检测类别**：
```
0: title    - 村落标题
1: caption  - 图注文字
2: txt_1    - 正文片段1
3: txt_2    - 正文片段2
4: img      - 插图图片
5: txt_3    - 正文片段3
6: txt_4    - 正文片段4
```

#### 2. Final_output/ - 最终输出结果
- `各村OCR结果/`：按 `序号_村名` 组织的每个村落目录
- `广州市从化区卷一_1-260.txt`：全书合并后的完整文本

#### 3. Temp_data/ - 中间结果目录
- `images_PDF/`：PDF 转换后图片及 YOLO 检测标注
- `images_cropped_villages/`：按标题分类裁剪后的图片及元数据
- `ocr_json_results/`：OCR 中间 JSON 结果

---

## 流水线说明

### 步骤 1：PDF 转图片
**文件**：`util/pdf_to_images.py`

功能：
- 交互式输入 PDF 文件路径
- 输入输出目录后自动将 PDF 每页转为300 DPI PNG图片
- 按 `Page_001.png`、`Page_002.png` 命名输出

> 运行后会提示输入源 PDF 和图片保存目录，执行完成后按回车退出。

### 步骤 2：YOLO 目标检测
**文件**：
- `pipelines/Step1_YOLO_detect/detect_pdf_yolo_xhao.py`

说明：
- `detect_pdf_yolo_xhao.py`：整合布局模型与正文模型，按固定类别顺序输出 YOLO 标注并生成可视化图

检测类别包含（见classes.txt）：
- `title`：村落标题
- `caption`：插图附文
- `txt_1`：正文片段1
- `txt_2`：正文片段2
- `img`：插图图片
- `txt_3`：正文片段3
- `txt_4`：正文片段4

### 步骤 3：智能分割归档
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

输出示例：
```
data/Temp_data/images_cropped_villages/
├── Page_022_title/
│   ├── Page_022_title.png
│   ├── Page_022_txt_1.png
│   ├── Page_022_img.png
│   ├── Page_022_caption.png
│   └── img_caption_metadata.json
```

### 步骤 4：OCR 识别与最终结果生成
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
### 依赖安装

#### 1. PaddleOCR环境

```bash
# 利用Anaconda，以下为经过验证的CPU版本安装命令
conda create -n paddleocr python=3.9 -y
conda activate paddleocr
python -m pip install --upgrade pip
python -m pip install paddleocr
pip install paddlepaddle==3.2.2 
pip install PyMuPDF
# GPU版本PaddlePaddle：pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
# 注意：一般情况下，PaddlePaddle需要先安装，再安装PaddleOCR
```

#### 2. YOLO环境
```bash
# 以下为AI所给命令，配置可参考B站视频
# 基础图像处理库
pip install opencv-python opencv-contrib-python
pip install ultralytics
```




### 安装验证
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

### 运行流程
1. 准备 PDF 文件
2. 运行 `python util/pdf_to_images.py`
3. 运行 `python pipelines/Step1_YOLO_detect/detect_pdf_yolo_xhao.py`
4. 运行 `python pipelines/Step2_Crop_by_YOLO_Label/crop_by_yolo_with_metadata.py`
5. 运行 `python process_cropped_data.py`

> 如果使用 GPU，请根据需要将脚本中的 `device` 参数修改为 `gpu`。

---

## 技术栈说明

| 组件 | 技术选型 | 用途 |
|------|---------|------|
| PDF 处理 | PyMuPDF (fitz) | PDF 转高清图片 |
| 目标检测 | YOLO11 / Ultralytics | 检测标题、正文、插图、图注 |
| OCR 识别 | PaddleOCR | 高精度中文 OCR |
| 图像处理 | OpenCV | 图片裁剪、可视化 |
