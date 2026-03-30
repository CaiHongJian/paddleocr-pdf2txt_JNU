# -*- coding: utf-8 -*-
"""
main.py
-功能：主程序入口，配置路径并调用 OCR 和 提取模块。
-作者：蔡鸿键
"""

import os
import ocr_module
import json2txt_module

def main():
    # --- 配置路径 ---
    INPUT_IMAGE_FOLDER = r"BookImages\allimages"           # 图片文件夹
    TEMP_JSON_FOLDER = r"OCR_IMGs2TXT_chj\ocr_out_jsons"          # 临时 JSON 文件夹（中间产物）
    OUTPUT_TEXT_FOLDER = r"OCR_IMGs2TXT_chj\ocr_out_texts"        # 最终 TXT 输出文件夹
    # ---------------------------------
    
    print(" 开始执行全流程任务...\n")

    # 运行 OCR 识别
    # 输入：图片文件夹 -> 输出：JSON 文件夹
    ocr_module.run_ocr(INPUT_IMAGE_FOLDER, TEMP_JSON_FOLDER)

    # 运行 JSON 转 TXT
    # 输入：JSON 文件夹 -> 输出：TXT 文件夹
    json2txt_module.extract_text(TEMP_JSON_FOLDER, OUTPUT_TEXT_FOLDER, indent_threshold=100)

    print("\n 全流程执行完毕！")

if __name__ == "__main__":
    main()