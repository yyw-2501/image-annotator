# -*- coding: utf-8 -*-
"""GUI 全流程模拟测试：构建窗口→拖入图片→点开始→等待完成信号→检查导出。"""
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import main
from PySide6.QtCore import QTimer

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE, "test_run", "big_photo.jpg")


def make_img():
    os.makedirs(os.path.join(BASE, "test_run"), exist_ok=True)
    if not os.path.exists(IMG):
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (4000, 3000), (200, 210, 220))
        ImageDraw.Draw(img).rectangle([500, 400, 3500, 2600], outline="red", width=30)
        img.save(IMG, quality=90)


class FakeMsgBox:
    @staticmethod
    def information(*a, **k):
        print("[MsgBox] information:", a[2] if len(a) > 2 else "")

    @staticmethod
    def warning(*a, **k):
        print("[MsgBox] warning:", a[2] if len(a) > 2 else "")


main.QMessageBox = FakeMsgBox

app = main.QApplication([])
w = main.MainWindow()
w.base_url_edit.setText("http://127.0.0.1:11434/v1")
w.refresh_models()
make_img()

print("模型数:", w.model_combo.count())
print("当前模型:", w.model_combo.currentData())

w.on_dropped([IMG])
print("列表图片数:", w.image_list.count())

for fmt, label in [("json", "out_json"), ("yolo", "out_yolo"),
                   ("coco", "out_coco"), ("markdown", "out_md")]:
    w.export_rows[fmt][0].setChecked(True)
    w.export_rows[fmt][1].setText(os.path.join(BASE, "test_run", label))

print("指令非空:", bool(w.prompt_edit.toPlainText().strip()))
w.start_batch()
print("runner 已启动:", w.runner is not None and w.runner.isRunning())


def check():
    if w.runner is None:
        print("RESULT: FAIL - runner 未创建")
        app.quit()
        return
    if w.runner.isFinished():
        w.runner.wait()
        print("RESULT: OK - 批处理结束")
        for fmt in ("json", "yolo", "coco", "markdown"):
            d = os.path.join(BASE, "test_run", "out_" + fmt if fmt != "markdown" else "out_md")
            files = sorted(os.listdir(d)) if os.path.isdir(d) else []
            print("  [%s] %s" % (fmt, ", ".join(files)))
        app.quit()
    else:
        QTimer.singleShot(1000, check)


QTimer.singleShot(2000, check)
app.exec()
