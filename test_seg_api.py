# -*- coding: utf-8 -*-
"""API 多边形模式端到端测试：mock VL 直接返回轮廓点 → 导出验证。"""
import os
import sys
import threading
import json
import http.server

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import main
from PySide6.QtCore import QTimer

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE, "test_run", "poly_test.png")


class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.dumps({"choices": [{"message": {"content": json.dumps({
            "description": "五边形目标",
            "objects": [{"name": "pentagon",
                         "points": [[100, 100], [200, 50], [300, 120], [260, 240], [120, 220]],
                         "confidence": 0.91}]})}}]}).encode()
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


srv = http.server.HTTPServer(("127.0.0.1", 18997), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()

os.makedirs(os.path.join(BASE, "test_run"), exist_ok=True)
from PIL import Image, ImageDraw
img = Image.new("RGB", (400, 300), (240, 240, 245))
ImageDraw.Draw(img).polygon([(100, 100), (200, 50), (300, 120), (260, 240), (120, 220)], fill=(60, 120, 60))
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
w.base_url_edit.setText("http://127.0.0.1:18997/v1")
w.refresh_models()
w.mode_combo.setCurrentIndex(1)  # 实例分割
w.seg_engine_combo.setCurrentIndex(0)  # API 多边形
w.on_mode_changed(None)
w.on_dropped([IMG])
w.export_rows["json"][0].setChecked(True)
w.export_rows["json"][1].setText(os.path.join(BASE, "test_run", "out_poly"))
w.export_rows["yolo"][0].setChecked(True)
w.export_rows["yolo"][1].setText(os.path.join(BASE, "test_run", "out_poly_yolo"))
w.start_batch()


def check():
    if w.runner is None or not w.runner.isFinished():
        QTimer.singleShot(1000, check)
        return
    w.runner.wait()
    j = json.load(open(os.path.join(BASE, "test_run", "out_poly", "poly_test.json"), encoding="utf-8"))
    b = j["boxes"][0]
    print("API多边形 bbox:", b["x1"], b["y1"], b["x2"], b["y2"])
    print("polygons 点数:", len(j["polygons"][0]))
    yolo = open(os.path.join(BASE, "test_run", "out_poly_yolo", "poly_test.txt"), encoding="utf-8").read().strip()
    print("YOLO-seg 行:", yolo[:60], "...")
    print("RESULT: OK")
    app.quit()


QTimer.singleShot(2000, check)
app.exec()
w.close()
srv.shutdown()