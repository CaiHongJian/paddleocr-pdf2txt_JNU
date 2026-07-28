# 任务目标：重构 main.py，实现一键全流程自动化

你是一名资深 Python 工程师，正在维护一个基于 PaddleOCR + YOLO 的《全粤村情》PDF 正文提取系统。
你的任务：**在不改动任何现有业务逻辑代码的前提下**，在工程根目录下新建 `main.py`，实现"一条龙"自动化流水线。

---

## ⚠️ 最高优先级约束（违反即失败）

1. **严禁修改以下目录中的任何业务代码**：
   - `util/` 下所有文件（含 `ocr_utils.py`、`pdf_to_images.py`、`txt_extractor.py`、`txt_merger.py`）
   - `pipelines/` 下所有文件（含 `detect_pdf_yolo_xhao.py`、`crop_by_yolo_with_metadata.py`）
   - `process_cropped_data.py`
   - OCR 相关逻辑**原样 import，不得重构、不得简化、不得"优化"**

2. **main.py 只做一件事：编排**。所有实际处理逻辑通过 import 调用现有模块完成。

3. **目录结构严格遵循项目现有约定**，不得自创路径：

```
data/
├── Final_output/              # 最终输出（main.py 唯一输出目标）
│   ├── 各村OCR结果/
│   └── *.txt
└── Temp_data/                 # 中间产物（全部自动管理）
    ├── images_PDF/            # PDF转图 + YOLO标注可视化
    ├── images_cropped_villages/  # 按村落裁剪结果 + img_caption_metadata.json
    └── ocr_json_results/      # OCR中间JSON结果
```

---

## 一、main.py 设计要求

### 1.1 配置区（必须在文件顶部，集中定义）

```python
# ========== 用户仅需修改以下两项 ==========
INPUT_PDF_PATH = r"path/to/input.pdf"          # 输入PDF路径
FINAL_OUTPUT_DIR = r"data/Final_output"        # 最终输出目录
# ==========================================

# 技术参数（必须保留中文注释，说明默认值选取理由）
PDF_DPI = 300                  # PDF转图分辨率；300DPI兼顾清晰度与性能，过低会导致OCR精度下降
YOLO_DEVICE = "auto"           # auto/gpu/cpu；auto自动检测CUDA可用性
YOLO_CONF_THRESH = 0.25        # YOLO置信度阈值；低于此值的目标被过滤，过高会漏检
YOLO_IOU_THRESH = 0.45         # NMS IOU阈值；控制重叠框合并强度
OCR_USE_GPU = True             # PaddleOCR是否使用GPU；False则强制CPU
OCR_MAX_BATCH_SIZE = 8         # OCR批处理大小；显存不足时可降至4或2
NUM_WORKERS = 4                # 并行worker数；CPU核心数的一半通常较优
```

### 1.2 环境信息打印（程序启动后第一行输出）

在 `main()` 最开始执行以下检查并打印，**格式整洁、一目了然**：

```text
================================================
  全粤村情 PDF → TXT 自动化处理系统
================================================
>>> 环境信息
  Python:          3.9.x
  PyTorch:         x.x.x
  CUDA 可用:       True / False
  当前设备:        GPU (CUDA) / CPU
  PaddlePaddle:    x.x.x
  PaddleOCR:       x.x.x
  OpenCV:          x.x.x
  PyMuPDF:         x.x.x
>>> 路径配置
  输入 PDF:        xxx.pdf
  输出目录:        data/Final_output
================================================
```

获取方式提示：
- PaddleOCR 版本：`from paddleocr import __version__ as paddleocr_version`
- PaddlePaddle 版本：`import paddle; paddle.__version__`
- CUDA 可用性：`torch.cuda.is_available()`

---

## 二、流水线步骤（与 README 完全一致）

main.py 按顺序编排以下 5 个阶段，**每个阶段对应一个 tqdm 进度条**，外加一个总进度条：

| 阶段 | 对应原脚本 | 说明 |
|------|-----------|------|
| Stage 1 | `util/pdf_to_images.py` | PDF → 高清 PNG |
| Stage 2 | `pipelines/Step1_YOLO_detect/detect_pdf_yolo_xhao.py` | YOLO 版面检测 |
| Stage 3 | `pipelines/Step2_Crop_by_YOLO_Label/crop_by_yolo_with_metadata.py` | 智能裁剪归档 |
| Stage 4 | `util/ocr_utils.py` | OCR 识别（批量） |
| Stage 5 | `process_cropped_data.py` 中的合并逻辑 | 文本合并 + 最终输出 |

### 2.1 tqdm 进度条设计

**总体要求：美观、信息密度适中、不刷屏**

```python
# 总进度条：5个阶段
with tqdm(total=5, desc="🚀 总体进度", position=0,
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar_total:

    # Stage 1
    with tqdm(total=num_pages, desc="📄 PDF转图", position=1, leave=False) as pbar:
        ...
        pbar.update(1)
    pbar_total.update(1)

    # Stage 2
    with tqdm(total=num_images, desc="🔍 YOLO检测", position=1, leave=False) as pbar:
        ...
    pbar_total.update(1)

    # Stage 3
    with tqdm(total=num_pages, desc="✂️  智能裁剪", position=1, leave=False) as pbar:
        ...
    pbar_total.update(1)

    # Stage 4
    with tqdm(total=num_crop_images, desc="📖 OCR识别", position=1, leave=False) as pbar:
        ...
    pbar_total.update(1)

    # Stage 5
    with tqdm(total=num_villages, desc="📝 文本合并", position=1, leave=False) as pbar:
        ...
    pbar_total.update(1)
```

**样式建议**：
- `position=0` 给总进度条，`position=1` 给子进度条，避免嵌套错乱
- `leave=False` 让子进度条完成后消失，保持终端干净
- 每个阶段完成后打印一行简短摘要（如 `✅ PDF转图完成：260页 → data/Temp_data/images_PDF/`）

---

## 三、各阶段集成方式（不得重写逻辑）

### Stage 1：PDF 转图

```python
from util.pdf_to_images import convert_pdf_to_images

# 调用原逻辑，仅传入路径参数
convert_pdf_to_images(
    pdf_path=INPUT_PDF_PATH,
    output_dir=Path("data/Temp_data/images_PDF"),
    dpi=PDF_DPI
)
```

### Stage 2：YOLO 检测

```python
from pipelines.Step1_YOLO_detect.detect_pdf_yolo_xhao import detect_layout_and_text

detect_layout_and_text(
    images_dir=Path("data/Temp_data/images_PDF"),
    output_dir=Path("data/Temp_data/images_PDF"),
    device=YOLO_DEVICE,
    conf_thresh=YOLO_CONF_THRESH,
    iou_thresh=YOLO_IOU_THRESH,
)
```

### Stage 3：智能裁剪归档

```python
from pipelines.Step2_Crop_by_YOLO_Label.crop_by_yolo_with_metadata import crop_by_yolo_metadata

crop_by_yolo_metadata(
    images_dir=Path("data/Temp_data/images_PDF"),
    labels_dir=Path("data/Temp_data/images_PDF"),
    output_dir=Path("data/Temp_data/images_cropped_villages"),
)
```

### Stage 4：OCR 识别

```python
from util.ocr_utils import batch_ocr_village_images

batch_ocr_village_images(
    cropped_dir=Path("data/Temp_data/images_cropped_villages"),
    output_dir=Path("data/Temp_data/ocr_json_results"),
    use_gpu=OCR_USE_GPU,
    batch_size=OCR_MAX_BATCH_SIZE,
    num_workers=NUM_WORKERS,
)
```

### Stage 5：文本合并与最终输出

```python
from util.txt_extractor import extract_text_with_indent
from util.txt_merger import merge_village_texts

# 提取带缩进文本
extract_text_with_indent(
    ocr_results_dir=Path("data/Temp_data/ocr_json_results"),
    output_base_dir=Path("data/Final_output/各村OCR结果"),
)

# 合并文本
merge_village_texts(
    villages_dir=Path("data/Final_output/各村OCR结果"),
    final_output_dir=Path("data/Final_output"),
)
```

> ⚠️ 以上函数签名如与实际不符，**以原文件为准**，main.py 适配原文件即可，不得修改原函数。

---

## 四、注释规范（必须严格执行）

### 4.1 文件头注释

```python
"""
全粤村情 PDF → TXT 自动化处理入口
================================

基于 PaddleOCR + YOLO 的双栏 PDF 正文提取系统一键流水线。
用户仅需配置 INPUT_PDF_PATH 和 FINAL_OUTPUT_DIR 即可完成全部处理。

流水线步骤：
  1. PDF → 高清 PNG 图片（300 DPI）
  2. YOLO 版面检测（标题/正文/插图/图注）
  3. 按村落标题智能裁剪归档（跨页归属 + 图注配对）
  4. PaddleOCR 批量文字识别
  5. 带缩进文本合并 + 按村落输出

Author: [你的名字]
Date:   2026-xx-xx
"""
```

### 4.2 函数 docstring 模板

```python
def stage1_pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int) -> None:
    """
    阶段1：将PDF文档批量转换为高清PNG图片

    参数:
        pdf_path:   输入PDF文件路径
        output_dir: 图片输出目录（data/Temp_data/images_PDF/）
        dpi:        输出分辨率，默认300；过低影响OCR精度，过高增加内存占用

    输出:
        Page_001.png, Page_002.png, ...

    依赖:
        util.pdf_to_images.convert_pdf_to_images
    """
```

### 4.3 技术参数注释要求

**每个参数必须说明三件事**：默认值是什么、为什么选这个值、调大/调小有什么影响。

```python
OCR_MAX_BATCH_SIZE = 8    # OCR批处理大小；GPU显存8GB建议8，4GB建议4，CPU建议2
YOLO_CONF_THRESH = 0.25   # 置信度阈值；降低可召回更多小目标，提高可减少误检
NUM_WORKERS = 4           # 并行数；设为CPU核心数一半，过高反而因GIL导致性能下降
```

### 4.4 关键步骤注释示例

```python
# ===== Stage 2: YOLO版面检测 =====
# 使用双模型串联：布局模型定位区域 → 正文模型精细检测
# 输出格式：YOLO标准标注 + 可视化叠加图
```

---

## 五、异常处理与日志

1. 每个阶段开始前检查前置依赖是否存在（如 Stage 2 检查 images_PDF 是否有图片）
2. 遇异常时打印清晰错误信息并退出，不吞异常
3. 使用 `print()` 即可，无需引入 logging 模块（保持轻量）
4. 关键节点输出摘要统计（页数、村数、识别耗时等）

---

## 六、执行步骤（AI 必须先思考再写代码）

- **Step 1**：阅读 README，确认所有目录路径、文件名、类名与上述一致
- **Step 2**：列出"受保护文件清单"，确认不会触碰任何业务代码
- **Step 3**：设计 tqdm 进度条层级结构（总进度条 + 5个子进度条）
- **Step 4**：编写 main.py，确保 import 路径正确
- **Step 5**：逐阶段检查：import 是否正确、参数是否传递、OCR 代码是否原样调用
- **Step 6**：自检确认清单（见下文）

---

## 七、交付前确认清单（✅ 必须全部打勾）

- [ ] main.py 位于工程根目录
- [ ] 配置区仅包含 INPUT_PDF_PATH 和 FINAL_OUTPUT_DIR 两个用户变量
- [ ] 环境信息在启动时完整打印（含 PaddleOCR / PaddlePaddle 版本、CPU / GPU 状态）
- [ ] 5 个阶段每个都有 tqdm 进度条 + 总进度条，显示层级正确、不刷屏
- [ ] 中间产物目录严格使用 `data/Temp_data/` 下三个子目录
- [ ] 最终输出目录严格使用 `data/Final_output/`
- [ ] 所有技术参数均有中文注释，说明默认值选取理由
- [ ] 每个函数均有中文 docstring（功能 / 输入 / 输出 / 依赖）
- [ ] `util/`、`pipelines/`、`process_cropped_data.py` 中的代码**零改动**
- [ ] OCR 相关逻辑原样 import，未做任何"优化"
- [ ] 代码可直接运行：`python main.py`

---

## 八、输出格式要求

请直接输出完整的 `main.py` 文件内容，用 ```` ```python ```` 代码块包裹。
开头简要说明你的设计思路（3~5 句话即可），然后给出完整代码。
不需要解释 README 已有的内容，不需要教我怎么安装依赖。
