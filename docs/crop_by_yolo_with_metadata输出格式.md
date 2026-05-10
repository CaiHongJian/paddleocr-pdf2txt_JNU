# crop_by_yolo_with_metadata 输出目录结构

## 根目录: Images_village_cropped/

```
Images_village_cropped/
├── Page_022_title/
│   ├── Page_022_caption.png
│   ├── Page_022_img.png
│   ├── Page_022_title.png
│   ├── Page_022_txt_1.png
│   ├── Page_022_txt_2.png
│   ├── ...
│   └── img_caption_metadata.json
│
├── Page_025_title/
│   ├── ...
│   └── img_caption_metadata.json
│
└── ...

```

## 结构说明

1. **按 title 分类归档**: 根目录下的每个子文件夹以 `Page_xxx_title` 格式命名，对应一个标题区域
2. **裁剪的图片类型**:
   - `*_title.png`: 标题裁剪图
   - `*_img.png`: 图片裁剪图
   - `*_caption.png`: 图注裁剪图
   - `*_txt_*.png`: 文本裁剪图（其他类别）
3. **元数据文件**: 每个 title 文件夹内都有 `img_caption_metadata.json`，记录该目录下所有 img-caption 配对的坐标信息
