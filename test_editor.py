# -*- coding: utf-8 -*-
"""标注修正器测试：加载记录 → 拖拽顶点 / 整体移动 / 加点删点 / 框角点修正 /
删除标注 / 类别名与置信度编辑 → 验证写回记录。"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication
from PIL import Image, ImageDraw

from editor import AnnotationEditor, AnnotationEditorWindow

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE, "test_run", "editor_test.png")


def make_img():
    os.makedirs(os.path.join(BASE, "test_run"), exist_ok=True)
    img = Image.new("RGB", (640, 480), (230, 235, 240))
    ImageDraw.Draw(img).rectangle([100, 100, 300, 250], fill=(150, 60, 60))
    img.save(IMG)


def mev(type_, pos, button=Qt.MouseButton.LeftButton,
        buttons=Qt.MouseButton.LeftButton):
    return QMouseEvent(type_, QPointF(pos[0], pos[1]), QPointF(pos[0], pos[1]),
                       button, buttons, Qt.KeyboardModifier.NoModifier)


def record_with(polygon, boxes):
    return {
        "image_path": IMG, "file_name": "editor_test.png",
        "width": 640, "height": 480, "description": "测试图",
        "status": "ok", "model": "mock", "timestamp": "now",
        "boxes": boxes,
        "polygons": polygon,
    }


def drag(ed, start, delta):
    ed.mousePressEvent(mev(QEvent.Type.MouseButtonPress, start))
    ed.mouseMoveEvent(mev(QEvent.Type.MouseMove, (start[0] + delta[0],
                                                  start[1] + delta[1])))
    ed.mouseReleaseEvent(mev(QEvent.Type.MouseButtonRelease,
                             (start[0] + delta[0], start[1] + delta[1])))


def test_editor():
    app = QApplication([])
    make_img()
    rec = record_with(
        [[[120.0, 120.0], [280.0, 130.0], [270.0, 240.0], [110.0, 230.0]]],
        [{"name": "car", "x1": 100.0, "y1": 100.0, "x2": 300.0, "y2": 250.0,
          "confidence": 0.9}],
    )
    ed = AnnotationEditor()
    ed.resize(800, 600)
    assert ed.set_record(rec), "载入记录失败"
    assert len(ed.shapes) == 1
    s = ed.shapes[0]
    assert s.has_polygon, "应加载为多边形"

    # 1. 拖拽顶点（视图偏移 20,10，换算回图像坐标）
    scale = ed._view_scale()
    v = ed._to_view(120.0, 120.0)
    drag(ed, v, (20, 10))
    p0 = s.polygon[0]
    assert abs(p0[0] - (120.0 + 20 / scale)) < 1.5 and \
           abs(p0[1] - (120.0 + 10 / scale)) < 1.5, "顶点拖拽失败: %s" % p0
    print("1. 顶点拖拽 OK ->", p0)

    # 2. 写回：多边形形状的框应同步为外接矩形
    ed.apply_to_record()
    b = rec["boxes"][0]
    assert b["x1"] == 110.0 and b["y1"] == 130.0 and b["x2"] == 280.0 and b["y2"] == 240.0, \
        "写回外接矩形失败: %s" % b
    print("2. 写回记录 OK ->", b)

    # 3. 双击边中点加点
    e0 = ed._to_view(s.polygon[0][0], s.polygon[0][1])
    e1 = ed._to_view(s.polygon[1][0], s.polygon[1][1])
    mid = ((e0[0] + e1[0]) / 2, (e0[1] + e1[1]) / 2)
    before = len(s.polygon)
    ed.mouseDoubleClickEvent(mev(QEvent.Type.MouseButtonDblClick, mid))
    assert len(s.polygon) == before + 1, "加点失败"
    print("3. 双击加点 OK ->", len(s.polygon), "点")

    # 4. 双击顶点删点（顶点与边中点可能太近，删除后重新取一个顶点）
    target = ed._to_view(s.polygon[2][0], s.polygon[2][1])
    ed.mouseDoubleClickEvent(mev(QEvent.Type.MouseButtonDblClick, target))
    assert len(s.polygon) == before, "删点失败"
    print("4. 双击删点 OK ->", len(s.polygon), "点")

    # 5. 整体移动（点四边形内部，避开顶点）
    cx = sum(p[0] for p in s.polygon) / 4
    cy = sum(p[1] for p in s.polygon) / 4
    iv = ed._to_view(cx, cy)
    before_pts = [list(p) for p in s.polygon]
    drag(ed, iv, (30, -20))
    for bp, p in zip(before_pts, s.polygon):
        assert abs(p[0] - bp[0] - 30 / scale) < 2.0 and \
               abs(p[1] - bp[1] + 20 / scale) < 2.0, "整体移动失败"
    print("5. 整体移动 OK")

    # 6. 纯框形状：拖拽角点修正，保持轴对齐
    rec2 = record_with([], [{"name": "box", "x1": 50.0, "y1": 60.0,
                             "x2": 200.0, "y2": 180.0, "confidence": 0.8}])
    assert ed.set_record(rec2), "载入纯框记录失败"
    s2 = ed.shapes[0]
    assert not s2.has_polygon
    c = ed._to_view(200.0, 60.0)  # 右上角
    drag(ed, c, (40, -30))
    x1, y1, x2, y2 = s2.box
    assert abs(x2 - (200.0 + 40 / scale)) < 1.5 and abs(y1 - (60.0 - 30 / scale)) < 1.5, \
        "框角点修正失败: %s" % s2.box
    assert abs(x1 - 50.0) < 0.6 and abs(y2 - 180.0) < 0.6, "轴对齐被破坏: %s" % s2.box
    ed.apply_to_record()
    assert rec2["boxes"][0]["x2"] == round(x2, 2) and rec2["boxes"][0]["y1"] == round(y1, 2)
    print("6. 框角点修正 OK ->", s2.box)

    # 7. Delete 键删除标注
    ed.select_shape(s2)
    ed.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete,
                               Qt.KeyboardModifier.NoModifier))
    assert len(ed.shapes) == 0, "Delete 删除失败"
    ed.apply_to_record()
    assert rec2["boxes"] == [] and rec2["polygons"] == [], "删除后写回失败"
    print("7. Delete 删除 OK")

    # 8. 修正窗口：列表、类别名编辑、导航
    win = AnnotationEditorWindow([rec, rec2], 0)
    win.show()
    assert win.shape_list.count() == 1, "窗口标注列表数量错误"
    win.on_list_clicked(win.shape_list.item(0))
    win.name_edit.setText("bus")
    win.conf_spin.setValue(0.42)
    assert rec["boxes"][0]["name"] == "bus", "类别名未写回"
    assert rec["boxes"][0]["confidence"] == 0.42, "置信度未写回"
    win.next_btn.click()
    assert win.title_label.text().startswith("2/2"), "下一张导航失败"
    win.prev_btn.click()
    assert win.title_label.text().startswith("1/2"), "上一张导航失败"
    win.close()
    print("8. 修正窗口 OK")

    # 9. 打点分割模式：收集正/负样本点 + 清除
    rec3 = record_with([], [])
    ed2 = AnnotationEditor()
    ed2.resize(800, 600)
    assert ed2.set_record(rec3), "载入空记录失败"
    ed2.set_sam_mode(True)
    ed2.mousePressEvent(mev(QEvent.Type.MouseButtonPress, ed2._to_view(200, 150)))
    ed2.mousePressEvent(mev(QEvent.Type.MouseButtonPress, ed2._to_view(300, 200),
                            button=Qt.MouseButton.RightButton, buttons=Qt.MouseButton.RightButton))
    pts, labels = ed2.get_sam_points()
    assert len(pts) == 2 and labels == [1, -1], "打点收集失败: %s %s" % (pts, labels)
    ed2.clear_sam_points()
    assert ed2.get_sam_points() == ([], []), "清除打点失败"
    ed2.set_sam_mode(False)
    print("9. 打点模式点收集/清除 OK ->", pts)

    # 10. 新建多边形：加点 + 双击闭合 + add_shape 派生外接矩形并写回
    ed2.set_poly_mode(True)
    for vx, vy in [(100, 100), (300, 100), (300, 300)]:
        ed2.mousePressEvent(mev(QEvent.Type.MouseButtonPress, ed2._to_view(vx, vy)))
    assert len(ed2._draft_poly) == 3, "多边形草稿加点失败"
    ed2.mouseDoubleClickEvent(mev(QEvent.Type.MouseButtonDblClick, ed2._to_view(100, 300)))
    assert ed2._draft_poly == [], "双击后草稿应清空"
    sh = ed2.add_shape("grape", [[100, 100], [300, 100], [300, 300], [100, 300]])
    assert sh.has_polygon and sh.box == [100, 100, 300, 300], "add_shape 派生外接矩形失败: %s" % sh.box
    ed2.apply_to_record()
    assert rec3["boxes"][0]["name"] == "grape" and len(rec3["polygons"][0]) == 4, "新建标注写回失败"
    print("10. 新建多边形 + add_shape 写回 OK ->", sh.box)

    # 11. 窗口层：打点/新建模式按钮联动 + 结果处理
    rec5 = record_with([], [])
    win2 = AnnotationEditorWindow([rec5], 0)
    win2.show()
    win2.sam_btn.setChecked(True)
    assert win2.editor._sam_mode and win2.run_sam_btn.isEnabled(), "打点模式联动失败"
    win2.poly_btn.setChecked(True)
    assert win2.editor._poly_mode and not win2.editor._sam_mode, "新建多边形模式联动失败"
    win2.on_sam_result(("ok", [[150, 150], [250, 150], [250, 250], [150, 250]]))
    assert len(win2.editor.shapes) == 1, "SAM 结果未新增标注"
    win2.on_polygon_created([[50, 50], [150, 50], [150, 150], [50, 150]])
    assert len(win2.editor.shapes) == 2, "新建多边形未新增标注"
    win2.close()
    print("11. 窗口层打点/新建多边形 OK")

    # 12. 完整异步打点分割流程（mock SAM 回调）
    def fake_sam(image_path, points, labels):
        return [[160, 160], [240, 160], [240, 240], [160, 240]]

    rec4 = record_with([], [])
    win3 = AnnotationEditorWindow([rec4], 0, sam_segmenter=fake_sam)
    win3.show()
    win3.sam_btn.setChecked(True)
    win3.editor.mousePressEvent(mev(QEvent.Type.MouseButtonPress, win3.editor._to_view(200, 200)))
    win3.on_run_sam()
    for _ in range(200):
        QApplication.processEvents()
        if len(win3.editor.shapes) == 1:
            break
        time.sleep(0.02)
    assert len(win3.editor.shapes) == 1, "异步 SAM 分割未新增标注"
    assert win3.editor.shapes[0].has_polygon, "异步分割结果应为多边形"
    win3.close()
    print("12. 异步打点分割 OK")

    print("RESULT: OK - 标注修正器测试全部通过")


if __name__ == "__main__":
    test_editor()
