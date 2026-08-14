# -*- coding: utf-8 -*-
"""端到端自测：生成测试图 → Ollama 推理 → 四种格式导出。"""
import os
import time

from PIL import Image, ImageDraw

from api_client import chat_json, DEFAULT_INSTRUCTION, SCHEMA_TEXT
from exporters import collect_images, EXPORTERS

BASE = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE, "test_run")


def make_test_images():
    os.makedirs(TEST_DIR, exist_ok=True)
    paths = []
    for k, color, shape in [(0, "red", "rect"), (1, "blue", "circle")]:
        img = Image.new("RGB", (640, 480), "white")
        d = ImageDraw.Draw(img)
        if shape == "rect":
            d.rectangle([80, 60, 560, 400], outline=color, width=8)
        else:
            d.ellipse([160, 90, 480, 390], outline=color, width=8)
        p = os.path.join(TEST_DIR, "test_%d.png" % k)
        img.save(p)
        paths.append(p)
    return paths


def main():
    # 通过环境变量配置后端，默认连本机 Ollama（可在分享给他人时按需覆盖）
    model = os.environ.get("ANNOTATOR_MODEL", "qwen3-vl:4b")
    base_url = os.environ.get("ANNOTATOR_BASE_URL", "http://127.0.0.1:11434/v1")
    api_key = os.environ.get("ANNOTATOR_API_KEY", "")
    api_config = {"base_url": base_url, "api_key": api_key}
    print("后端: %s（模型 %s）" % (base_url, model))
    paths = make_test_images()
    prompt = "%s\n\n%s" % (DEFAULT_INSTRUCTION, SCHEMA_TEXT)

    records = []
    for p in paths:
        t0 = time.time()
        try:
            res = chat_json(model, prompt, p, temperature=0.1, max_tokens=4096,
                            retries=3, log=print, api_config=api_config)
            print("推理 %s 用时 %.1fs -> boxes=%d, desc=%s"
                  % (os.path.basename(p), time.time() - t0, len(res["boxes"]),
                     res["description"][:40]))
            rec = {
                "image_path": os.path.normpath(p),
                "file_name": os.path.basename(p),
                "width": 640, "height": 480,
                "description": res["description"], "boxes": res["boxes"],
                "model": model,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "ok", "raw_response": "",
            }
        except Exception as e:
            print("推理失败 %s: %s" % (p, e))
            rec = {"image_path": os.path.normpath(p), "file_name": os.path.basename(p),
                   "width": 640, "height": 480, "description": "", "boxes": [],
                   "model": model, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "status": "error", "error": str(e), "raw_response": ""}
        records.append(rec)

    for fmt, (label, func) in EXPORTERS.items():
        out = os.path.join(TEST_DIR, "out_" + fmt)
        n = func(records, out, log=print)
        files = sorted(os.listdir(out))
        print("[%s] 导出 %d 文件 → %s: %s" % (label, n, out, ", ".join(files)))

    print("collect_images 测试:", collect_images([TEST_DIR]))


if __name__ == "__main__":
    main()
