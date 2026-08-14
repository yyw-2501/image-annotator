# -*- coding: utf-8 -*-
"""交互式标注修正器：在图上查看并手动修正自动标注结果。

操作方式：
    - 左键拖拽白色控制点：微调轮廓顶点 / 检测框角点
    - 双击多边形边：插入顶点；双击顶点：删除顶点（至少保留 3 点）
    - 拖拽标注主体（多边形内部 / 框内）：整体移动
    - 中键拖动画布平移，滚轮缩放
    - Delete 键 / 按钮：删除选中标注
    - 右侧列表点击选择标注；类别名与置信度可随时修改

坐标一律使用原图像素；画布按 fit + zoom + pan 换算显示。
修正结果实时写回对应 record（原地修改），主程序导出时自动使用最新数据。
"""
import math
import os
import threading

from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QSplitter,
    QVBoxLayout, QWidget,
)

HANDLE_R = 4.5      # 选中控制点半径（视图像素）
HIT_R = 8.0         # 命中判定半径（视图像素）
ZOOM_MIN, ZOOM_MAX = 0.1, 20.0

SHAPE_COLORS = [
    (255, 99, 99), (89, 200, 89), (99, 150, 255), (255, 200, 60),
    (220, 120, 255), (80, 210, 210), (255, 140, 60), (170, 170, 170),
]

CANVAS_BG = QColor(32, 32, 36)


def fit_to_screen(win, width, height):
    """按主屏可用区域自适应窗口大小：小屏自动缩小到屏幕 92%，避免显示不全。"""
    screen = QApplication.primaryScreen()
    if screen is not None:
        g = screen.availableGeometry()
        if g.width() > 200 and g.height() > 200:
            width = min(width, int(g.width() * 0.92))
            height = min(height, int(g.height() * 0.92))
    win.resize(width, height)


def _color_for(i):
    r, g, b = SHAPE_COLORS[i % len(SHAPE_COLORS)]
    return QColor(r, g, b)


class Shape:
    """一条标注。box=[x1,y1,x2,y2] 与 polygon=[[x,y],...] 可并存（SAM2 模式）
    或二选一（检测 / API 多边形模式），坐标均为原图像素。"""

    def __init__(self, name, confidence, box=None, polygon=None, color=None):
        self.name = name
        self.confidence = float(confidence)
        self.box = box
        self.polygon = polygon
        self.color = color

    @property
    def has_polygon(self):
        return bool(self.polygon and len(self.polygon) >= 3)

    def polygon_bbox(self):
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return [min(xs), min(ys), max(xs), max(ys)]

    def box_points(self):
        x1, y1, x2, y2 = self.box
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _point_in_polygon(p, poly):
    """射线法：判断点 p（QPointF）是否在多边形 poly（QPointF 列表）内。"""
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i].x(), poly[i].y()
        xj, yj = poly[j].x(), poly[j].y()
        if (yi > p.y()) != (yj > p.y()):
            denom = (yj - yi) or 1e-9
            if p.x() < (xj - xi) * (p.y() - yi) / denom + xi:
                inside = not inside
        j = i
    return inside


def _dist_seg(p, a, b):
    """点 p 到线段 ab 的距离（均为 QPointF）。"""
    dx, dy = b.x() - a.x(), b.y() - a.y()
    if dx == 0 and dy == 0:
        return math.hypot(p.x() - a.x(), p.y() - a.y())
    t = ((p.x() - a.x()) * dx + (p.y() - a.y()) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(p.x() - (a.x() + t * dx), p.y() - (a.y() + t * dy))


def _poly_bbox(poly):
    """多边形外接矩形 [x1, y1, x2, y2]。"""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


class AnnotationEditor(QWidget):
    """可交互的画布：显示原图 + 标注，支持点修正、加点删点、整体移动、缩放平移。"""

    edited = Signal()                    # 有几何/属性被修改
    selected_changed = Signal(object)    # 选中形状变化（Shape 或 None）
    sam_points_changed = Signal()        # 打点分割的点集变化
    polygon_created = Signal(object)     # 手动新建多边形完成（顶点列表）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.shapes = []
        self.selected = None
        self._rec = None
        self._pixmap = None
        self._img_w = self._img_h = 0
        self._fit_scale = 1.0
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._drag = None    # ("point", shape, idx) | ("shape", shape, pos) | ("pan", pos)
        self._sam_mode = False
        self._sam_points = []    # 图像坐标 [[x, y], ...]
        self._sam_labels = []    # [1 / -1, ...]
        self._poly_mode = False
        self._draft_poly = []    # 图像坐标 [[x, y], ...]
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------- 视图坐标换算
    def _view_scale(self):
        return self._fit_scale * self._zoom

    def _view_origin(self):
        x = self._pan.x() + (self.width() - self._img_w * self._fit_scale) / 2.0
        y = self._pan.y() + (self.height() - self._img_h * self._fit_scale) / 2.0
        return x, y

    def _to_view(self, ix, iy):
        x0, y0 = self._view_origin()
        s = self._view_scale()
        return x0 + ix * s, y0 + iy * s

    def _to_img(self, vx, vy):
        x0, y0 = self._view_origin()
        s = self._view_scale()
        return (vx - x0) / s, (vy - y0) / s

    # ------------------------------------------------- 数据载入 / 写回
    def clear(self):
        self.shapes = []
        self.selected = None
        self._rec = None
        self._pixmap = None
        self._img_w = self._img_h = 0
        self._zoom, self._pan = 1.0, QPointF(0, 0)
        self._sam_mode = False
        self._sam_points = []
        self._sam_labels = []
        self._poly_mode = False
        self._draft_poly = []
        self.update()

    def set_record(self, rec):
        """载入一条标注记录；无可用数据或图片无法读取时返回 False。"""
        self.clear()
        if not rec or not os.path.isfile(rec.get("image_path", "")):
            return False
        pix = QPixmap(rec["image_path"])
        if pix.isNull():
            return False
        self._rec = rec
        self._pixmap = pix
        self._img_w, self._img_h = pix.width(), pix.height()
        boxes = rec.get("boxes") or []
        polys = rec.get("polygons") or []
        ok_polys = polys if len(polys) == len(boxes) else []
        for i, b in enumerate(boxes):
            poly = None
            if i < len(ok_polys) and len(ok_polys[i]) >= 3:
                poly = [list(p) for p in ok_polys[i]]
            self.shapes.append(Shape(
                b["name"], b.get("confidence", 1.0),
                box=[b["x1"], b["y1"], b["x2"], b["y2"]],
                polygon=poly, color=_color_for(i),
            ))
        self.update()
        return True

    def apply_to_record(self):
        """把修正结果写回记录（原地修改）。多边形形状的框同步为其外接矩形。"""
        rec = self._rec
        if rec is None:
            return
        boxes, polys = [], []
        for s in self.shapes:
            if s.has_polygon:
                s.box = s.polygon_bbox()
            bx = s.box or [0, 0, 0, 0]
            boxes.append({
                "name": s.name,
                "confidence": round(min(max(float(s.confidence), 0.0), 1.0), 4),
                "x1": round(bx[0], 2), "y1": round(bx[1], 2),
                "x2": round(bx[2], 2), "y2": round(bx[3], 2),
            })
            polys.append([[round(x, 1), round(y, 1)] for x, y in (s.polygon or [])])
        rec["boxes"] = boxes
        rec["polygons"] = polys

    def set_sam_mode(self, on):
        self._sam_mode = on
        if on:
            self._poly_mode = False
        if not on:
            self._sam_points = []
            self._sam_labels = []
        self.update()

    def set_poly_mode(self, on):
        self._poly_mode = on
        if on:
            self._sam_mode = False
        if not on:
            self._draft_poly = []
        self.update()

    def get_sam_points(self):
        return list(self._sam_points), list(self._sam_labels)

    def clear_sam_points(self):
        self._sam_points = []
        self._sam_labels = []
        self.sam_points_changed.emit()
        self.update()

    def add_shape(self, name, polygon, confidence=1.0):
        """新增一条标注（打点分割结果 / 手动多边形），自动派生外接矩形。"""
        poly = [list(p) for p in polygon] if polygon else None
        box = _poly_bbox(poly) if poly and len(poly) >= 3 else None
        sh = Shape(name, confidence, box=box, polygon=poly, color=_color_for(len(self.shapes)))
        self.shapes.append(sh)
        self._select(sh)
        self.edited.emit()
        self.update()
        return sh

    def select_shape(self, shape):
        self._select(shape)

    def delete_selected(self):
        if self.selected is not None and self.selected in self.shapes:
            self.shapes.remove(self.selected)
            self._select(None)
            self.edited.emit()
            self.update()

    def _select(self, shape):
        if shape is not self.selected:
            self.selected = shape
            self.selected_changed.emit(shape)
        self.update()

    # ------------------------------------------------- 命中测试
    def _shape_points(self, s):
        return s.polygon if s.has_polygon else s.box_points()

    def _hit_handle(self, vp):
        """命中控制点，返回 (shape, 顶点下标)。"""
        for s in self.shapes:
            for i, (ix, iy) in enumerate(self._shape_points(s)):
                vx, vy = self._to_view(ix, iy)
                if abs(vx - vp.x()) <= HIT_R and abs(vy - vp.y()) <= HIT_R:
                    return s, i
        return None

    def _hit_shape(self, vp):
        """命中标注主体（多边形内部 / 框内），后绘制的优先。"""
        for s in reversed(self.shapes):
            if s.has_polygon:
                pts = [QPointF(*self._to_view(x, y)) for x, y in s.polygon]
                if _point_in_polygon(vp, pts):
                    return s
            else:
                x1, y1 = self._to_view(s.box[0], s.box[1])
                x2, y2 = self._to_view(s.box[2], s.box[3])
                if QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)).contains(vp):
                    return s
        return None

    def _hit_edge(self, vp, s):
        """命中多边形边，返回边起点下标（含闭合边）。"""
        pts = [QPointF(*self._to_view(x, y)) for x, y in s.polygon]
        for i in range(len(pts)):
            if _dist_seg(vp, pts[i - 1], pts[i]) <= HIT_R:
                return i - 1
        return None

    # ------------------------------------------------- 鼠标 / 键盘交互
    def mousePressEvent(self, e):
        if self._pixmap is None:
            return
        self.setFocus()
        pos = e.position()
        if e.button() == Qt.MouseButton.MiddleButton:
            self._drag = ("pan", pos)
            return
        # 打点分割模式：左键加正样本点，右键加负样本点
        if self._sam_mode:
            if e.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
                ix, iy = self._to_img(pos.x(), pos.y())
                ix = min(max(ix, 0.0), self._img_w)
                iy = min(max(iy, 0.0), self._img_h)
                self._sam_points.append([round(ix, 1), round(iy, 1)])
                self._sam_labels.append(1 if e.button() == Qt.MouseButton.LeftButton else -1)
                self.sam_points_changed.emit()
                self.update()
            return
        # 新建多边形模式：左键加点，右键撤销
        if self._poly_mode:
            if e.button() == Qt.MouseButton.LeftButton:
                ix, iy = self._to_img(pos.x(), pos.y())
                ix = min(max(ix, 0.0), self._img_w)
                iy = min(max(iy, 0.0), self._img_h)
                self._draft_poly.append([round(ix, 1), round(iy, 1)])
                self.update()
            elif e.button() == Qt.MouseButton.RightButton and self._draft_poly:
                self._draft_poly.pop()
                self.update()
            return
        if e.button() != Qt.MouseButton.LeftButton:
            return
        hit = self._hit_handle(pos)
        if hit is not None:
            s, i = hit
            self._select(s)
            self._drag = ("point", s, i)
            return
        s = self._hit_shape(pos)
        if s is not None:
            self._select(s)
            self._drag = ("shape", s, pos)
            return
        self._select(None)

    def mouseMoveEvent(self, e):
        if self._drag is None or self._pixmap is None:
            return
        kind = self._drag[0]
        pos = e.position()
        if kind == "pan":
            d = pos - self._drag[1]
            self._pan = QPointF(self._pan.x() + d.x(), self._pan.y() + d.y())
            self._drag = ("pan", pos)
            self.update()
            return
        if kind == "point":
            _, s, i = self._drag
            ix, iy = self._to_img(pos.x(), pos.y())
            ix = min(max(ix, 0.0), self._img_w)
            iy = min(max(iy, 0.0), self._img_h)
            if s.has_polygon:
                s.polygon[i][0] = round(ix, 1)
                s.polygon[i][1] = round(iy, 1)
            else:
                pts = s.box_points()
                pts[i] = [ix, iy]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                s.box = [min(xs), min(ys), max(xs), max(ys)]
            self.edited.emit()
            self.update()
            return
        # 整体移动
        _, s, start = self._drag
        d = pos - start
        self._drag = ("shape", s, pos)
        dx, dy = d.x() / self._view_scale(), d.y() / self._view_scale()
        if s.has_polygon:
            for p in s.polygon:
                p[0] = round(min(max(p[0] + dx, 0.0), self._img_w), 1)
                p[1] = round(min(max(p[1] + dy, 0.0), self._img_h), 1)
        else:
            x1, y1, x2, y2 = s.box
            nx1, ny1 = x1 + dx, y1 + dy
            nx2, ny2 = x2 + dx, y2 + dy
            if nx1 < 0:
                nx1, nx2 = 0, nx2 - nx1
            if nx2 > self._img_w:
                nx1, nx2 = nx1 - (nx2 - self._img_w), self._img_w
            if ny1 < 0:
                ny1, ny2 = 0, ny2 - ny1
            if ny2 > self._img_h:
                ny1, ny2 = ny1 - (ny2 - self._img_h), self._img_h
            s.box = [round(nx1, 1), round(ny1, 1), round(nx2, 1), round(ny2, 1)]
        self.edited.emit()
        self.update()

    def mouseReleaseEvent(self, e):
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        if self._pixmap is None:
            return
        # 新建多边形模式：双击闭合完成
        if self._poly_mode and e.button() == Qt.MouseButton.LeftButton:
            if len(self._draft_poly) >= 3:
                self.polygon_created.emit(list(self._draft_poly))
            self._draft_poly = []
            self.update()
            return
        if e.button() != Qt.MouseButton.LeftButton:
            return
        pos = e.position()
        hit = self._hit_handle(pos)
        if hit is not None:
            s, i = hit
            if s.has_polygon and len(s.polygon) > 3:
                del s.polygon[i]
                self.edited.emit()
                self.update()
            return
        s = self._hit_shape(pos)
        if s is not None and s.has_polygon:
            edge = self._hit_edge(pos, s)
            if edge is not None:
                ix, iy = self._to_img(pos.x(), pos.y())
                s.polygon.insert(edge + 1, [round(ix, 1), round(iy, 1)])
                self._select(s)
                self.edited.emit()
                self.update()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
        else:
            super().keyPressEvent(e)

    def wheelEvent(self, e):
        if self._pixmap is None:
            return
        factor = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        new_zoom = min(max(self._zoom * factor, ZOOM_MIN), ZOOM_MAX)
        cursor_img = self._to_img(e.position().x(), e.position().y())
        self._zoom = new_zoom
        vx, vy = self._to_view(*cursor_img)
        self._pan = QPointF(self._pan.x() + (e.position().x() - vx),
                            self._pan.y() + (e.position().y() - vy))
        self.update()

    def resizeEvent(self, e):
        if self._img_w and self._img_h:
            self._fit_scale = min(self.width() / self._img_w,
                                  self.height() / self._img_h)
        super().resizeEvent(e)

    # ------------------------------------------------- 绘制
    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), CANVAS_BG)
        if self._pixmap is None:
            p.setPen(QColor(120, 120, 120))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "双击左侧列表中的图片载入标注进行修正")
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        x0, y0 = self._view_origin()
        s = self._view_scale()
        p.drawPixmap(QRectF(x0, y0, self._img_w * s, self._img_h * s),
                     self._pixmap, QRectF(0, 0, self._img_w, self._img_h))
        for idx, sh in enumerate(self.shapes):
            self._draw_shape(p, sh)
        # 打点模式：绘制已收集的提示点
        if self._sam_mode:
            for (px, py), lab in zip(self._sam_points, self._sam_labels):
                vx, vy = self._to_view(px, py)
                c = QColor(0, 255, 0) if lab > 0 else QColor(255, 60, 60)
                p.setBrush(c)
                p.setPen(QPen(c, 2))
                p.drawEllipse(QPointF(vx, vy), 5, 5)
                p.setPen(QColor(255, 255, 255))
                p.drawText(int(vx) + 7, int(vy) - 7, "+" if lab > 0 else "-")
        # 新建多边形模式：绘制草稿折线
        if self._poly_mode and self._draft_poly:
            pts = [QPointF(*self._to_view(x, y)) for x, y in self._draft_poly]
            if len(pts) >= 2:
                path = QPainterPath()
                path.moveTo(pts[0])
                for pt in pts[1:]:
                    path.lineTo(pt)
                p.setPen(QPen(QColor(255, 255, 0), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(path)
            for pt in pts:
                p.setPen(QPen(QColor(255, 255, 0), 2))
                p.setBrush(QColor(255, 255, 0))
                p.drawEllipse(pt, 4, 4)

    def _draw_shape(self, p, sh):
        sel = sh is self.selected
        color = sh.color or QColor(255, 255, 255)
        pen = QPen(color, 2.5 if sel else 1.5)
        if sh.has_polygon:
            pts = [QPointF(*self._to_view(x, y)) for x, y in sh.polygon]
            path = QPainterPath()
            path.moveTo(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            path.closeSubpath()
            fill = QColor(color)
            fill.setAlpha(55 if sel else 35)
            p.fillPath(path, fill)
            p.setPen(pen)
            p.drawPath(path)
        if sh.box:
            x1, y1 = self._to_view(sh.box[0], sh.box[1])
            x2, y2 = self._to_view(sh.box[2], sh.box[3])
            p.setPen(pen)
            p.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
        ax = sh.box[0] if sh.box else sh.polygon[0][0]
        ay = sh.box[1] if sh.box else sh.polygon[0][1]
        vx, vy = self._to_view(ax, ay)
        p.setPen(QColor(255, 255, 255))
        font = p.font()
        font.setPointSize(9)
        p.setFont(font)
        p.drawText(int(vx) + 5, int(vy) - 6,
                   "%s %.2f" % (sh.name, sh.confidence))
        if sel:
            for px, py in self._shape_points(sh):
                hx, hy = self._to_view(px, py)
                p.setBrush(QColor(255, 255, 255))
                p.setPen(QPen(color, 2))
                p.drawEllipse(QPointF(hx, hy), HANDLE_R, HANDLE_R)


class AnnotationEditorWindow(QMainWindow):
    """标注修正主窗口：画布 + 标注列表 + 类别/置信度编辑 + 上一张/下一张导航，
    支持打点交互式重跑 SAM 分割与手动新建多边形补漏标。

    records 为共享列表（主程序 self.records），修正实时写回其中的记录。
    sam_segmenter 为可选回调 (image_path, points, labels) -> polygon，用于打点分割。
    """

    _sam_result = Signal(object)   # ("ok", polygon) | ("err", msg)

    def __init__(self, records, index=0, parent=None, sam_segmenter=None):
        super().__init__(parent)
        self.setWindowTitle("标注修正")
        fit_to_screen(self, 1120, 760)
        self.records = [r for r in records
                        if r.get("status") == "ok" and os.path.isfile(r.get("image_path", ""))]
        self.index = 0
        if self.records and 0 <= index < len(records) and records[index] in self.records:
            self.index = self.records.index(records[index])
        self._sel_shape = None
        self._sam_segmenter = sam_segmenter
        self._sam_result.connect(self.on_sam_result)

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ---- 顶部工具行 1：导航 + 保存
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("◀ 上一张")
        self.prev_btn.clicked.connect(lambda: self._jump(-1))
        self.next_btn = QPushButton("下一张 ▶")
        self.next_btn.clicked.connect(lambda: self._jump(1))
        self.title_label = QLabel("")
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color:#888;")
        save_btn = QPushButton("✓ 保存并关闭")
        save_btn.clicked.connect(self.close)
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        nav.addWidget(self.title_label, 1)
        nav.addWidget(self.status_label)
        nav.addWidget(save_btn)
        root.addLayout(nav)

        # ---- 顶部工具行 2：选中标注属性
        attr = QHBoxLayout()
        attr.addWidget(QLabel("选中标注:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("类别名")
        self.name_edit.setEnabled(False)
        self.name_edit.textChanged.connect(self.on_name_changed)
        attr.addWidget(self.name_edit, 1)
        attr.addWidget(QLabel("置信度:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setEnabled(False)
        self.conf_spin.valueChanged.connect(self.on_conf_changed)
        attr.addWidget(self.conf_spin)
        self.delete_btn = QPushButton("删除选中标注")
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self.on_delete_clicked)
        attr.addWidget(self.delete_btn)
        root.addLayout(attr)

        # ---- 顶部工具行 3：打点分割 + 新建多边形
        tool = QHBoxLayout()
        self.sam_btn = QPushButton("🔍 打点分割")
        self.sam_btn.setCheckable(True)
        self.sam_btn.toggled.connect(self.on_sam_mode)
        self.run_sam_btn = QPushButton("▶ 执行分割")
        self.run_sam_btn.setEnabled(False)
        self.run_sam_btn.clicked.connect(self.on_run_sam)
        self.clear_pts_btn = QPushButton("✕ 清除点")
        self.clear_pts_btn.setEnabled(False)
        self.clear_pts_btn.clicked.connect(self.on_clear_points)
        self.poly_btn = QPushButton("➕ 新建多边形")
        self.poly_btn.setCheckable(True)
        self.poly_btn.toggled.connect(self.on_poly_mode)
        tool.addWidget(self.sam_btn)
        tool.addWidget(self.run_sam_btn)
        tool.addWidget(self.clear_pts_btn)
        tool.addWidget(self.poly_btn)
        tool.addStretch(1)
        root.addLayout(tool)

        # ---- 主体：画布 + 右侧列表
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.editor = AnnotationEditor()
        self.editor.edited.connect(self.on_edited)
        self.editor.selected_changed.connect(self.on_selected_changed)
        self.editor.sam_points_changed.connect(self.on_points_changed)
        self.editor.polygon_created.connect(self.on_polygon_created)
        splitter.addWidget(self.editor)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(QLabel("标注列表（点击选中）"))
        self.shape_list = QListWidget()
        self.shape_list.itemClicked.connect(self.on_list_clicked)
        rl.addWidget(self.shape_list, 1)
        rl.addWidget(QLabel("模型描述:"))
        self.desc_label = QLabel("")
        self.desc_label.setWordWrap(True)
        rl.addWidget(self.desc_label)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        hint = QLabel("提示：拖动白点修正轮廓/框角 · 双击轮廓边加点 · 双击顶点删点 · "
                      "拖标注主体整体移动 · 中键平移画布 · 滚轮缩放 · Delete 删除标注")
        hint.setStyleSheet("color:#888;")
        hint.setWordWrap(True)
        root.addWidget(hint)

        self._load(self.index)

    # ------------------------------------------------- 导航 / 载入
    def _jump(self, delta):
        self._apply()
        target = min(max(self.index + delta, 0), len(self.records) - 1)
        if target != self.index:
            self.index = target
            self._load(target)

    def _load(self, index):
        if not self.records:
            self.title_label.setText("无可用标注（请先运行批量标注）")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.editor.clear()
            return
        rec = self.records[index]
        ok = self.editor.set_record(rec)
        self._apply()
        if not ok:
            QMessageBox.warning(self, "提示", "无法读取图片：%s" % rec.get("image_path", ""))
        self.title_label.setText("%d/%d  %s" % (index + 1, len(self.records), rec["file_name"]))
        self.desc_label.setText(rec.get("description") or "（无描述）")
        self.prev_btn.setEnabled(index > 0)
        self.next_btn.setEnabled(index < len(self.records) - 1)
        self._sel_shape = None
        self.status_label.setText("")
        self.sam_btn.setChecked(False)
        self.poly_btn.setChecked(False)
        self.refresh_list()
        self.update_attr_controls()

    def _apply(self):
        self.editor.apply_to_record()

    # ------------------------------------------------- 编辑同步
    def on_edited(self):
        self._apply()
        self.status_label.setText("已修正（实时生效）")
        self.refresh_list()

    def on_selected_changed(self, shape):
        self._sel_shape = shape
        self.update_attr_controls()

    def update_attr_controls(self):
        sh = self._sel_shape
        self.name_edit.setEnabled(sh is not None)
        self.conf_spin.setEnabled(sh is not None)
        self.delete_btn.setEnabled(sh is not None)
        if sh is not None:
            self.name_edit.blockSignals(True)
            self.conf_spin.blockSignals(True)
            self.name_edit.setText(sh.name)
            self.conf_spin.setValue(sh.confidence)
            self.name_edit.blockSignals(False)
            self.conf_spin.blockSignals(False)

    def on_name_changed(self, text):
        if self._sel_shape is not None:
            self._sel_shape.name = text.strip() or "unknown"
            self.on_edited()

    def on_conf_changed(self, value):
        if self._sel_shape is not None:
            self._sel_shape.confidence = value
            self.on_edited()

    def on_delete_clicked(self):
        self.editor.delete_selected()

    # ------------------------------------------------- 打点分割 / 新建多边形
    def on_sam_mode(self, on):
        self.editor.set_sam_mode(on)
        if on:
            self.poly_btn.setChecked(False)
            self.status_label.setText("打点模式：左键=正样本点，右键=负样本点，点“执行分割”")
        self.run_sam_btn.setEnabled(on)
        self.clear_pts_btn.setEnabled(on)

    def on_poly_mode(self, on):
        self.editor.set_poly_mode(on)
        if on:
            self.sam_btn.setChecked(False)
            self.status_label.setText("新建多边形：左键加点，右键撤销，双击闭合完成")

    def on_clear_points(self):
        self.editor.clear_sam_points()
        self.status_label.setText("已清除打点")

    def on_points_changed(self):
        pts, _ = self.editor.get_sam_points()
        self.status_label.setText("已打 %d 个点（左键+ / 右键-），点“执行分割”" % len(pts))

    def on_run_sam(self):
        points, labels = self.editor.get_sam_points()
        if not points:
            QMessageBox.information(self, "提示", "请先在图上打点（左键正样本 / 右键负样本）。")
            return
        if self._sam_segmenter is None:
            QMessageBox.warning(self, "提示", "当前未配置 SAM2 环境，无法打点分割。")
            return
        image_path = self.records[self.index].get("image_path", "")
        self.run_sam_btn.setEnabled(False)
        self.status_label.setText("SAM 分割中…")

        def worker():
            try:
                poly = self._sam_segmenter(image_path, points, labels)
                self._sam_result.emit(("ok", poly))
            except Exception as e:  # noqa: BLE001
                self._sam_result.emit(("err", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def on_sam_result(self, result):
        self.run_sam_btn.setEnabled(True)
        status, payload = result
        if status == "err":
            self.status_label.setText("")
            QMessageBox.warning(self, "分割失败", payload)
            return
        poly = payload
        if not poly or len(poly) < 3:
            self.status_label.setText("未分割出目标，请调整打点位置后重试")
            return
        self.editor.add_shape("sam", poly)
        self.status_label.setText("已新增分割结果，可在左侧改类别名")
        self.editor.clear_sam_points()
        self.refresh_list()

    def on_polygon_created(self, poly):
        self.editor.add_shape("new", poly)
        self.status_label.setText("已新建多边形标注，可在左侧改类别名")
        self.refresh_list()

    def on_list_clicked(self, item):
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is not None and 0 <= idx < len(self.editor.shapes):
            self.editor.select_shape(self.editor.shapes[idx])

    def refresh_list(self):
        self.shape_list.blockSignals(True)
        self.shape_list.clear()
        for i, sh in enumerate(self.editor.shapes):
            item = QListWidgetItem("%d. %s (%.2f)" % (i + 1, sh.name, sh.confidence))
            item.setData(Qt.ItemDataRole.UserRole, i)
            c = sh.color or QColor(255, 255, 255)
            item.setBackground(c.darker(130))
            self.shape_list.addItem(item)
        if self.editor.selected in self.editor.shapes:
            self.shape_list.setCurrentRow(self.editor.shapes.index(self.editor.selected))
        self.shape_list.blockSignals(False)
