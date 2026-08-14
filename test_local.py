# -*- coding: utf-8 -*-
"""本地 YOLO 模型标注测试：真实权重推理 + GUI 本地引擎全流程。

权重通过环境变量 ANNOTATOR_LOCAL_WEIGHTS 指定；缺省用本机已有的 yolo11n-seg.pt。
"""
import json
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import main
from PySide6.QtCore import QTimer

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE, "test_run", "local_test.png")
WEIGHTS = os.environ.get("ANNOTATOR_LOCAL_WEIGHTS",
                         r"D:\葡萄\moxing\10777647\yolo11n-seg.pt")


def make_img():
    from PIL import Image, ImageDraw
    os.makedirs(os.path.join(BASE, "test_run"), exist_ok=True)
    img = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([120, 120, 360, 360], fill=(230, 120, 20))
    d.rectangle([400, 300, 580, 440], fill=(40, 80, 200))
    img.save(IMG, quality=90)
    return IMG


class FakeMsgBox:
    @staticmethod
    def information(*a, **k):
        print("[MsgBox]", a[2] if len(a) > 2 else "")

    @staticmethod
    def warning(*a, **k):
        print("[MsgBox]", a[2] if len(a) > 2 else "")


main.QMessageBox = FakeMsgBox


def test_worker():
    """真实 LocalWorker 推理：检测 + 分割。"""
    make_img()
    py = main.find_local_python()
    assert py, "未找到 ultralytics 环境"
    print("本地环境:", py)
    w = main.LocalWorker(py, WEIGHTS, log=print)
    objs = w.infer(IMG, 0.25)
    print("检测目标数:", len(objs))
    assert len(objs) >= 1, "本地模型未检测到目标"
    for o in objs:
        print("  %s conf=%.3f box=%s points=%d"
              % (o["name"], o["confidence"],
                 [o["x1"], o["y1"], o["x2"], o["y2"]], len(o.get("points", []))))
    w.close()
    return objs


def test_gui():
    """GUI 本地引擎全流程：推理 + JSON/YOLO 导出。"""
    make_img()
    app = main.QApplication([])
    w = main.MainWindow()
    w.engine_combo.setCurrentIndex(1)  # 本地模型
    w.weights_edit.setText(WEIGHTS)
    w.local_python_edit.setText(main.find_local_python())
    w.local_conf_spin.setValue(0.25)
    w.on_dropped([IMG])
    w.export_rows["json"][0].setChecked(True)
    w.export_rows["json"][1].setText(os.path.join(BASE, "test_run", "out_local_json"))
    w.export_rows["yolo"][0].setChecked(True)
    w.export_rows["yolo"][1].setText(os.path.join(BASE, "test_run", "out_local_yolo"))
    w.start_batch()
    print("runner 启动:", w.runner is not None and w.runner.isRunning())

    def check():
        if w.runner is None or not w.runner.isFinished():
            QTimer.singleShot(1000, check)
            return
        w.runner.wait()
        j = os.path.join(BASE, "test_run", "out_local_json", "local_test.json")
        data = json.load(open(j, encoding="utf-8"))
        print("JSON boxes:", len(data["boxes"]), "polygons:", len(data.get("polygons", [])))
        assert len(data["boxes"]) >= 1, "本地模型全流程未产出标注"
        print("RESULT: OK - 本地模型全流程通过")
        app.quit()

    QTimer.singleShot(2000, check)
    app.exec()
    w.close()


if __name__ == "__main__":
    test_worker()
    test_gui()
