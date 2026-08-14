# -*- coding: utf-8 -*-
"""停止按钮专项测试：批处理中点击停止，应立即中断在途推理并结束。"""
import os
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import main
from PySide6.QtCore import QTimer

BASE = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE, "test_run")
T0 = time.time()
STOP_TIME = None
FINISH_TIME = None
DONE_ITEMS = 0


def make_imgs():
    from PIL import Image, ImageDraw
    os.makedirs(TEST_DIR, exist_ok=True)
    for k in range(2):
        p = os.path.join(TEST_DIR, "stop_%d.png" % k)
        if not os.path.exists(p):
            img = Image.new("RGB", (640, 480), "white")
            ImageDraw.Draw(img).rectangle([80, 60, 560, 400], outline="red", width=8)
            img.save(p)


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
w.base_url_edit.setText("http://127.0.0.1:11434/v1")
w.refresh_models()
make_imgs()
imgs = [os.path.join(TEST_DIR, "stop_%d.png" % k) for k in range(2)]
w.on_dropped(imgs)
w.export_rows["json"][0].setChecked(True)
w.export_rows["json"][1].setText(os.path.join(TEST_DIR, "out_json"))

orig_done = w.on_item_done


def on_done(info):
    global DONE_ITEMS, STOP_TIME
    orig_done(info)
    DONE_ITEMS += 1
    if DONE_ITEMS == 1 and STOP_TIME is None:
        STOP_TIME = time.time()
        print(">>> 第1张完成(%.1fs)，立即点击停止" % (STOP_TIME - T0))
        w.stop_batch()


w.on_item_done = on_done

orig_finished = w.on_finished


def on_finished(summary):
    global FINISH_TIME
    orig_finished(summary)
    FINISH_TIME = time.time()
    print(">>> 已结束，停止后耗时 %.2fs" % (FINISH_TIME - STOP_TIME))
    print("RESULT: %s" % ("PASS" if (FINISH_TIME - STOP_TIME) < 10 else "FAIL"))
    app.quit()


w.on_finished = on_finished
w.start_batch()
QTimer.singleShot(120000, lambda: (print("RESULT: FAIL - 120s 未结束"), app.quit()))
app.exec()
