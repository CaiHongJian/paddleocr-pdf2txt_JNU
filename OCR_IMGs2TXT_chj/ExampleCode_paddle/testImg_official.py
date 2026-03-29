# 官方推荐的使用方式，推荐使用 PaddleOCR 类来进行 OCR 识别，
# PaddleOCR 类会自动加载文本检测、文本识别、文本方向分类和文本图像矫正模型，
# 并且提供了更为简洁的接口来进行 OCR 识别。


from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
    use_doc_unwarping=False,            # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
    use_textline_orientation=False,     # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
    # device="gpu",                       # 通过 device 参数使得在模型推理时使用 GPU
)
# ocr = PaddleOCR(lang="en") # 通过 lang 参数来使用英文模型
# ocr = PaddleOCR(ocr_version="PP-OCRv4") # 通过 ocr_version 参数来使用 PP-OCR 其他版本
# ocr = PaddleOCR(device="gpu") # 通过 device 参数使得在模型推理时使用 GPU
# ocr = PaddleOCR(
#     text_detection_model_name="PP-OCRv5_server_det",
#     text_recognition_model_name="PP-OCRv5_server_rec",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
# ) # 更换 PP-OCRv5_server 模型
result = ocr.predict(r"test.png")
for res in result:
    res.print()
    res.save_to_img(r"output")
    res.save_to_json(r"output")