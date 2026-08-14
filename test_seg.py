# -*- coding: utf-8 -*-
"""SAM2 全链路端到端测试：mock VL API（返回 bbox）→ 真实 SAM2 分割 → 四格式导出验证。
运行于 base 环境；SAM2 环境通过 ANNOTATOR_SAM_PYTHON 指定，缺省自动探测。"""
import os
import sys
import threading
import json
import http.server

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import main
from PySide6.QtCore import QTimer

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE, "test_run", "seg_test.png")
OUT = os.path.join(BASE, "test_run", "out_seg")

# ---- 1. mock VL API：返回两个 bbox（椭圆+矩形）----
MOCK_BOXES = [{"name": "berry", "x1": 140, "y1": 90, "x2": 500, "y2": 390, "confidence": 0.95},
              {"name": "leaf", "x1": 20, "y1": 20, "x2": 200, "y2": 200, "confidence": 0.9}]


class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.dumps({"choices": [{"message": {"content": json.dumps(
            {"description": "图中有一个红色椭圆和一个蓝色矩形", "boxes": MOCK_BOXES})}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        body = json.dumps({"data": [{"id": "mock-vl"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


srv = http.server.HTTPServer(("127.0.0.1", 18998), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()

# ---- 2. 构造测试图 ----
os.makedirs(os.path.join(BASE, "test_run"), exist_ok=True)
from PIL import Image, ImageDraw
img = Image.new("RGB", (640, 480), (240, 240, 245))
d = ImageDraw.Draw(img)
d.ellipse([140, 90, 500, 390], fill=(180, 40, 40))
d.rectangle([20, 20, 200, 200], fill=(40, 40, 180))
img.save(IMG)


class FakeMsgBox:
    @staticmethod
    def information(*a, **k):
        print("[MsgBox]", a[2] if len(a) > 2 else "")

    @staticmethod
    def warning(*a, **k):
        print("[MsgBox]", a[2] if len(a) > 2 else "")


main.QMessageBox = FakeMsgBox
app = main.QApplication([])
w = main.MainWindow()
w.base_url_edit.setText("http://127.0.0.1:18998/v1")
w.refresh_models()

# 切到实例分割 + SAM2
w.mode_combo.setCurrentIndex(1)  # 实例分割
w.seg_engine_combo.setCurrentIndex(1)  # 本地 SAM2
_sam_py = os.environ.get("ANNOTATOR_SAM_PYTHON") or main.find_sam2_python()
assert _sam_py, "未找到 SAM2 环境（可设置 ANNOTATOR_SAM_PYTHON 环境变量）"
w.sam_python_edit.setText(_sam_py)
w.on_mode_changed(None)
assert w.seg_engine_combo.isEnabled() and w.sam_python_edit.isEnabled(), "控件联动失败"

w.on_dropped([IMG])
for fmt, label in [("json", "out_seg_json"), ("yolo", "out_seg_yolo"),
                   ("coco", "out_seg_coco"), ("markdown", "out_seg_md")]:
    w.export_rows[fmt][0].setChecked(True)
    w.export_rows[fmt][1].setText(os.path.join(BASE, "test_run", label))

w.start_batch()
print("runner 启动:", w.runner is not None and w.runner.isRunning())


def check():
    if w.runner is None or not w.runner.isFinished():
        QTimer.singleShot(1000, check)
        return
    w.runner.wait()
    print("RESULT: OK - 批处理结束")
    yolo_txt = os.path.join(BASE, "test_run", "out_seg_yolo",
                            "seg_test.txt")
    if os.path.exists(yolo_txt):
        line = open(yolo_txt, encoding="utf-8").read().strip()
        print("YOLO-seg 行数:", len(line.splitlines()))
        for ln in line.splitlines():
            parts = ln.split()
            print("  类别 %s, 点对 %d" % (parts[0], (len(parts) - 5) // 2))
    coco = json.load(open(os.path.join(BASE, "test_run", "out_seg_coco", "annotations.json"), encoding="utf-8"))
    for ann in coco["annotations"]:
        print("COCO segmentation 点数:", len(ann["segmentation"][0]) // 2)
    j = json.load(open(os.path.join(BASE, "test_run", "out_seg_json", "seg_test.json"), encoding="utf-8"))
    print("JSON polygons 数:", len(j["polygons"]))
    app.quit()


QTimer.singleShot(2000, check)
app.exec()
w.close()
srv.shutdown()
