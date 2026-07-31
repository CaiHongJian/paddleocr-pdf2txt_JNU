# paddleocr-pdf2txt_JNU

基于 PaddleOCR 与 YOLO 的双栏 PDF 正文提取系统，专门用于处理《全粤村情》文档。系统实现从 PDF 解析、目标检测、裁剪归档、OCR 识别到按村落输出的最终自动化流水线。

## 项目功能概述

本项目实现了从 PDF 文档中自动提取各村村落信息的完整流程：

1. **PDF 转图片**：将 PDF 文档批量转换为高清 JPG 图片（300 DPI）
2. **YOLO 目标检测**：识别页面中的标题、正文、插图、图注等元素
3. **智能分割归档**：根据检测结果裁剪图片，并按村落标题归属自动分类
4. **OCR 文字识别**：使用 PaddleOCR 对裁剪图片进行文字识别
5. **智能文本合并**：根据缩进规则判断段落结构，合并同一村落的完整文本
6. **插图重命名**：识别图注文字并作为插图文件名保存

> **一键运行**：配置 `main.py` 顶部的 `INPUT_PDF_PATH` 后执行 `python main.py` 即可完成全部步骤，无需分步操作。

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
│   │   └── <PDF文件名>.txt      # 全书合并文本（动态命名）
│   └── Temp_data/               # 临时数据目录（DEBUG=0 时自动清理）
│       ├── images_PDF/          # 存放PDF转换后的JPG图片及YOLO标注
│       ├── images_cropped_villages/   # 按村落标题分类裁剪的结果
│       └── ocr_json_results/          # OCR中间JSON结果
├── docs/                        # 文档目录
│   ├── 汇报记录/
│   ├── crop_by_yolo_with_metadata输出格式.md
│   └── 各村OCR结果文件结构.md
├── models/                      # 模型文件目录
│   ├── Model_xhao/
│   │   ├── detect_layout.pt     # 版面布局检测模型
│   │   └── detect_text.pt       # 文本区域检测模型
│   ├── Village_Model_chj/       # （废弃）
│   └── yolo11n.pt
├── pipelines/
│   ├── Step1_YOLO_detect/
│   │   └── detect_pdf_yolo_xhao.py
│   └── Step2_Crop_by_YOLO_Label/
│       └── crop_by_yolo_with_metadata.py
├── util/
│   ├── ocr_utils.py             # PaddleOCR 封装（线程安全）
│   ├── pdf_to_images.py         # 独立PDF转图工具（交互式）
│   ├── txt_extractor.py         # 带缩进的文本提取
│   └── txt_merger.py            # 智能文本合并
├── main.py                      # ★ 一键全流程入口（自动化流水线）
├── process_cropped_data.py      # 独立OCR处理脚本（可单独运行）
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
- `<PDF文件名>.txt`：全书合并后的完整文本（根据输入 PDF 文件名自动命名）

#### 3. Temp_data/ - 中间结果目录
- `images_PDF/`：PDF 转换后 JPG 图片及 YOLO 检测标注
- `images_cropped_villages/`：按标题分类裁剪后的 JPG 图片及元数据
- `ocr_json_results/`：OCR 中间 JSON 结果
- **DEBUG=0 时此目录在流水线结束后自动清理**

---

## 一键流水线（main.py）

### 快速开始

1. 打开 `main.py`，修改顶部配置：
   ```python
   INPUT_PDF_PATH = r"D:/path/to/your.pdf"   # 输入PDF路径
   FINAL_OUTPUT_DIR = r"data/Final_output"   # 输出目录（通常无需修改）
   ```
2. 运行：
   ```bash
   python main.py
   ```

### 配置参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `INPUT_PDF_PATH` | — | **必填**，输入 PDF 文件路径 |
| `FINAL_OUTPUT_DIR` | `data/Final_output` | 最终输出目录 |
| `DEBUG` | `0` | 调试模式：`1`=保留临时文件，`0`=自动清理 |
| `PDF_DPI` | `300` | PDF 转图分辨率 |
| `YOLO_DEVICE` | `auto` | YOLO 推理设备：`auto`/`gpu`/`cpu` |
| `YOLO_CONF_THRESH` | `0.25` | YOLO 置信度阈值 |
| `YOLO_IOU_THRESH` | `0.45` | NMS IOU 阈值 |
| `OCR_USE_GPU` | `False` | PaddleOCR是否使用GPU ，`True`代表使用GPU|
| `NUM_WORKERS` | `4` | OCR 线程池并行数 |
| `PIPELINE_WORKERS` | `2` | Stage 2 YOLO 流水线并行数 |

### 流水线架构

```
┌──────────────────────────────────────────────────────────┐
│                    main.py 一键流水线                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Stage 1+2  PDF转JPG ──→ YOLO检测    [生产者-消费者并行]  │
│      │  fitz 渲染         │  双模型推理                   │
│      │  Page_001.jpg      │  labels/*.txt                 │
│      ↓                    ↓                               │
│  Stage 3    智能裁剪归档               [串行]              │
│      │  两次扫描 + 跨页归属 + 图注配对                     │
│      ↓  images_cropped_villages/*.jpg                    │
│                                                          │
│  Stage 4    PaddleOCR 批量识别        [ThreadPoolExecutor] │
│      │  每线程独立OCR实例 + 小尺寸图自动padding            │
│      ↓  ocr_json_results/*.json                          │
│                                                          │
│  Stage 5    文本合并 + 村落输出       [串行]              │
│      │  缩进段落判断 + 页码排序合并                        │
│      ↓  各村OCR结果/ + <PDF名>.txt                       │
│                                                          │
│  DEBUG=0 → 自动清理 Temp_data/                           │
└──────────────────────────────────────────────────────────┘
```

### 异步并行处理

| 阶段 | 并行策略 | 说明 |
|------|----------|------|
| Stage 1→2 | 生产者-消费者队列 | PDF 转图与 YOLO 检测流水线并行，产出一张图立即检测 |
| Stage 4 | ThreadPoolExecutor | 多线程 OCR 识别，每线程独立 PaddleOCR 实例（`threading.local`） |

### DEBUG 模式

- **`DEBUG = 1`**：保留所有中间产物（`images_PDF/`、`images_cropped_villages/`、`ocr_json_results/`），便于调试和排查问题
- **`DEBUG = 0`**：流水线完成后自动清理 `Temp_data/` 目录，仅保留 `Final_output/` 中的最终结果

---

## 流水线步骤详解

### 步骤 1：PDF 转图片
**文件**：`util/pdf_to_images.py`（独立工具）/ `main.py` Stage 1（集成版）

功能：
- 将 PDF 每页转为 300 DPI JPG 图片
- 按 `Page_001.jpg`、`Page_002.jpg` 命名输出

> `util/pdf_to_images.py` 为交互式独立工具，`main.py` 中集成了参数化自动调用。

### 步骤 2：YOLO 目标检测
**文件**：`pipelines/Step1_YOLO_detect/detect_pdf_yolo_xhao.py`

说明：
- 整合布局模型与正文模型，按固定类别顺序输出 YOLO 标注并生成可视化图
- 在 `main.py` 中与 Stage 1 流水线并行执行，预融合模型（`fuse()`）+ 推理锁保证多线程安全

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
   - 第一遍扫描：收集所有页面的 title 坐标信息，建立全局 title 索引
   - 第二遍扫描：根据归属关系将每个元素分配到最近的 title 下

2. **跨页归属逻辑**：
   - 某个 title 的作用域从其出现位置开始，直到下一个 title 出现前（跨页有效）

3. **插图-图注配对**：
   - 规则：每个 caption 匹配其上方最近的 img（y_center 在 caption 下方且距离最小）
   - 生成 `img_caption_metadata.json` 记录配对信息

输出示例：
```
data/Temp_data/images_cropped_villages/
├── Page_022_title/
│   ├── Page_022_title.jpg
│   ├── Page_022_txt_1.jpg
│   ├── Page_022_img.jpg
│   ├── Page_022_caption.jpg
│   └── img_caption_metadata.json
```

### 步骤 4：OCR 识别
**文件**：`util/ocr_utils.py` + `main.py` Stage 4

处理流程：
1. **多线程并行**：使用 `ThreadPoolExecutor` 并发识别裁剪图片
2. **线程安全**：每个线程通过 `threading.local()` 创建独立的 PaddleOCR 实例
3. **小尺寸图片处理**：对小于 32×32 像素的图片自动 padding 白边至最小要求
4. **失败重试**：单张图片 OCR 失败后自动重试 1 次

### 步骤 5：智能文本合并与最终输出
**文件**：`util/txt_extractor.py`、`util/txt_merger.py` + `main.py` Stage 5

处理流程：
1. **识别村落名称**：对每个 title 图片进行 OCR，得到真实村名
2. **识别图注并重命名插图**：利用元数据配对关系，识别 caption 文字作为插图文件名
3. **带缩进的文本提取**：
   - 根据文本框的 X 坐标判断是否为缩进段落
   - 缩进达到阈值的行前面添加制表符 `\t`
4. **智能文本合并**：
   - 按页码和 txt 编号排序所有文本片段
   - 根据片段首行是否有制表符 `\t` 决定是否换行
5. **合并总文件**：所有村落文本合并为一个 `<PDF文件名>.txt`

最终输出目录结构：
```
data/Final_output/
├── 各村OCR结果/
│   ├── 1_大围村/
│   │   ├── 1_大围村.txt
│   │   ├── 大围村村貌（摄于2017年，从化区档案局提供）.jpg
│   │   └── ...
│   ├── 2_沙岗村/
│   │   └── ...
│   └── ...
└── <PDF文件名>.txt    # 全书合并文本
```

---

## 环境配置

### 依赖安装

#### 1. PaddleOCR 环境

```bash
# 利用Anaconda。
# 为避免兼容问题，paddleocr版本建议下载3.4.0版本，paddlepaddle根据实际情况下载。

# CPU版本paddleocr安装配置命令
conda create -n ocr python=3.9 -y
conda activate ocr
python -m pip install --upgrade pip
pip install paddlepaddle==3.2.2
python -m pip install paddleocr==3.4.0

# 对于GPU版本，paddleocr同样安装3.4.0版本。
# PaddlePaddle gpu版安装命令可查看：https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/windows-pip.html
# 以下为经过Python 3.9.23, Windows 10, CUDA 11.8, 1080Ti GPU版本验证过的安装命令
conda create -n ocr python=3.9 -y
conda activate ocr
pip install PyMuPDF tqdm paddleocr
pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install aiohttp==3.8.6
```

#### 2. YOLO 环境
```bash
conda activate ocr
# CPU 版本
pip3 install torch torchvision
pip install -U ultralytics
# GPU 版本 pytorch 安装见 https://pytorch.org/
```

#### 3. 其他依赖
```bash
conda activate ocr
pip install PyMuPDF tqdm
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

### 方式一：一键运行（推荐）

```bash
conda activate ocr

# 1. 修改 main.py 顶部的 INPUT_PDF_PATH
# 2. 如需保留中间文件调试，将 DEBUG 改为 1
# 3. 如使用 GPU，将 OCR_USE_GPU 改为 True
# 4. 运行
python main.py
```

### 方式二：分步运行（调试用）

```bash
python util/pdf_to_images.py
python pipelines/Step1_YOLO_detect/detect_pdf_yolo_xhao.py
python pipelines/Step2_Crop_by_YOLO_Label/crop_by_yolo_with_metadata.py
python process_cropped_data.py
```

> 如果使用 GPU，请根据需要将脚本中的 `device` 参数修改为 `gpu`，`main.py` 中将 `OCR_USE_GPU` 改为 `True`。

---

## 技术栈说明

| 组件 | 技术选型 | 用途 |
|------|---------|------|
| PDF 处理 | PyMuPDF (fitz) | PDF 转高清 JPG 图片 |
| 目标检测 | YOLO11 / Ultralytics | 检测标题、正文、插图、图注 |
| OCR 识别 | PaddleOCR (PP-OCRv5) | 高精度中文 OCR |
| 图像处理 | OpenCV | 图片裁剪、可视化 |
| 并行处理 | threading + ThreadPoolExecutor | 流水线并行 + 线程池并行 |
| 进度显示 | tqdm | 多阶段进度条 |
