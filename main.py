# -*- coding: utf-8 -*-
"""图像自动标注工具 — 桌面 GUI 入口。

功能：批量拖拽图片 → OpenAI 兼容视觉模型按自定义指令推理（目标检测 / 实例分割，
      实例分割支持 API 轮廓直出或本地 SAM2 像素分割）→
      结果自动导出；双击结果打开交互式修正窗口，手动拖拽/增删轮廓点后重新导出。
"""
import glob
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from PySide6.QtCore import Qt, QSize, QSettings, Signal, QThread
from PySide6.QtGui import QIcon, QImageReader
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPlainTextEdit, QListWidget, QListWidgetItem,
    QCheckBox, QPushButton, QProgressBar, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFileDialog, QMessageBox, QSplitter, QFrame,
)

from api_client import (
    ApiError, ApiCancel, list_vision_models, chat_json, chat_polygons,
    DEFAULT_INSTRUCTION, SCHEMA_TEXT, SCHEMA_TEXT_POLY,
)
from editor import AnnotationEditorWindow, fit_to_screen
from exporters import collect_images, EXPORTERS

FORMAT_LABELS = {
    "json": "JSON（每图一个 .json）",
    "yolo": "YOLO TXT（每图 .txt + classes.txt）",
    "coco": "COCO（单文件 annotations.json）",
    "markdown": "Markdown（图文描述 .md，图片自动复制）",
    "voc": "Pascal VOC XML（每图 .xml）",
    "labelme": "LabelMe JSON（每图多边形 .json）",
    "csv": "CSV 汇总（annotations.csv）",
    "mask": "语义分割掩码 PNG（类别着色）",
}


class ImageListWidget(QListWidget):
    """支持多选图片 / 文件夹拖拽的列表。"""
    dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(False)
        self.setIconSize(QSize(72, 72))
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setSpacing(2)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.dropped.emit(paths)


class SamWorker:
    """SAM2 常驻子进程客户端（子进程运行于装有 torch+sam2 的 conda 环境）。

    协议见 sam_cli.py：stdin/stdout 逐行 JSON 通信，分割请求串行执行。
    """

    def __init__(self, python_path, log=None):
        self.python = python_path
        self.log = log or (lambda s: None)
        self.proc = None
        self._lock = threading.Lock()
        self._start()

    def _start(self):
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam_cli.py")
        self.proc = subprocess.Popen(
            [self.python, script, "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def _alive(self):
        return self.proc is not None and self.proc.poll() is None

    def segment(self, image, boxes):
        """对图片的每个框做实例分割，返回多边形列表（与 boxes 一一对应）。"""
        with self._lock:
            if not self._alive():
                self.log("SAM2 子进程已退出，重新启动…")
                self._start()
            task = json.dumps({"image": image, "boxes": boxes})
            try:
                self.proc.stdin.write(task + "\n")
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()
            except (OSError, ValueError) as e:
                raise ApiError("SAM2 子进程通信失败: %s" % e)
            if not line:
                raise ApiError("SAM2 子进程无响应（可能环境缺失 torch/sam2）")
            res = json.loads(line)
            if res.get("error"):
                raise ApiError("SAM2 分割失败: %s" % res["error"])
            return res.get("polygons") or []

    def segment_points(self, image, points, labels=None):
        """用点提示（正/负样本）做分割，返回多边形列表（通常单元素）。"""
        with self._lock:
            if not self._alive():
                self.log("SAM2 子进程已退出，重新启动…")
                self._start()
            task = json.dumps({"image": image, "points": points,
                               "labels": labels or [1] * len(points)})
            try:
                self.proc.stdin.write(task + "\n")
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()
            except (OSError, ValueError) as e:
                raise ApiError("SAM2 子进程通信失败: %s" % e)
            if not line:
                raise ApiError("SAM2 子进程无响应（可能环境缺失 torch/sam2）")
            res = json.loads(line)
            if res.get("error"):
                raise ApiError("SAM2 分割失败: %s" % res["error"])
            return res.get("polygons") or []

    def close(self):
        with self._lock:
            if self.proc is not None:
                try:
                    self.proc.terminate()
                except Exception:
                    pass
                self.proc = None


# ---------------------------------------------------------------- SAM2 环境探测
def _candidate_conda_envs():
    """收集候选 conda 环境目录（含 base），来源：conda 命令 / 常见安装位置 / 环境变量。"""
    dirs = []
    # 1) 优先用 conda 命令（最准确）；不可用时静默降级
    try:
        proc = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True, text=True, timeout=20)
        if proc.returncode == 0:
            data = json.loads(proc.stdout or "{}")
            dirs.extend(p for p in data.get("envs", []) if p and os.path.isdir(p))
    except Exception:
        pass

    # 2) 常见安装根目录（base）+ 其 envs/ 子目录
    roots = []
    home = os.path.expanduser("~")
    base_names = ("anaconda3", "miniconda3", "Anaconda3", "Miniconda3",
                  "anaconda", "miniconda")
    for name in base_names:
        roots.append(os.path.join(home, name))
    if sys.platform.startswith("win"):
        for drive in ("C:\\", "D:\\", "E:\\"):
            for name in base_names:
                roots.append(drive + name)
            roots.append(drive + "ProgramData\\anaconda3")
            roots.append(drive + "ProgramData\\miniconda3")
    else:
        roots += ["/opt/anaconda3", "/opt/miniconda3", "/usr/local/anaconda3",
                  "/usr/local/miniconda3", "/opt/conda"]

    # 3) 环境变量
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        roots.append(os.path.dirname(conda_exe))
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        roots.append(conda_prefix)

    seen = set()
    for root in roots:
        root = os.path.normpath(root)
        if not root or root in seen or not os.path.isdir(root):
            continue
        seen.add(root)
        dirs.append(root)  # base 本身
        envs_dir = os.path.join(root, "envs")
        if os.path.isdir(envs_dir):
            for name in sorted(os.listdir(envs_dir)):
                child = os.path.join(envs_dir, name)
                if os.path.isdir(child):
                    dirs.append(child)

    # 去重保序
    result, seen2 = [], set()
    for d in dirs:
        d = os.path.normpath(d)
        if d not in seen2:
            seen2.add(d)
            result.append(d)
    return result


def _env_python_exe(env_dir):
    if sys.platform.startswith("win"):
        p = os.path.join(env_dir, "python.exe")
    else:
        p = os.path.join(env_dir, "bin", "python")
    return p if os.path.isfile(p) else None


def _env_has_sam2(env_dir):
    """通过文件系统快速判断环境是否已安装 sam2（不启动解释器）。"""
    if sys.platform.startswith("win"):
        return os.path.isdir(os.path.join(env_dir, "Lib", "site-packages", "sam2"))
    for sp in glob.glob(os.path.join(env_dir, "lib", "python*", "site-packages")):
        if os.path.isdir(os.path.join(sp, "sam2")):
            return True
    return False


def find_sam2_python():
    """自动探测装有 sam2 的 conda 环境 Python；找到返回路径，否则返回 None。"""
    for env_dir in _candidate_conda_envs():
        if _env_has_sam2(env_dir):
            py = _env_python_exe(env_dir)
            if py:
                return py
    return None


class BatchRunner(QThread):
    """后台批量推理 + 导出线程。"""
    sig_item_done = Signal(dict)
    sig_finished = Signal(dict)
    sig_log = Signal(str)

    def __init__(self, images, model, instruction, temperature, max_tokens,
                 concurrency, export_config, max_side=1600, parent=None,
                 mode="detect", seg_engine=None, sam_worker=None):
        super().__init__(parent)
        self.images = images
        self.model = model
        self.instruction = instruction
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        self.export_config = export_config  # {"json": {"enabled": bool, "dir": str}, ...}
        self.max_side = max_side  # 推理前图片最长边缩放上限(px)，0=不缩放
        self.mode = mode  # "detect" | "segment"
        self.seg_engine = seg_engine  # "api" | "sam2"
        self.sam_worker = sam_worker
        self._stop = threading.Event()
        self._abort_regs = []  # 在途请求的中止器列表（线程安全）
        self._reg_lock = threading.Lock()

    def stop(self):
        """立即停止：断开所有在途 HTTP 连接，Ollama 检测到断开会马上取消当前推理。"""
        self._stop.set()
        with self._reg_lock:
            aborts = list(self._abort_regs)
        for abort in aborts:
            try:
                abort()
            except Exception:
                pass

    @staticmethod
    def _prepare_image(path, max_side):
        """返回 (推理用临时文件路径, 原图宽, 原图高, 缩放比 scale)。
        scale = 推理图尺寸 / 原图尺寸，用于把模型输出的像素坐标换算回原图。"""
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        size = reader.size()
        width, height = size.width(), size.height()
        if width <= 0 or height <= 0:
            raise ValueError("无法读取图片尺寸（文件可能已损坏或格式不支持）")
        if max_side and max(width, height) > max_side:
            ratio = max_side / max(width, height)
            reader.setScaledSize(QSize(max(1, int(width * ratio)),
                                       max(1, int(height * ratio))))
            img = reader.read()
            if img.isNull():
                raise ValueError("图片缩放读取失败")
            tmp = os.path.join(tempfile.gettempdir(),
                               "annotator_%d_%s.jpg" % (os.getpid(), os.path.basename(path)))
            if not img.save(tmp, "JPEG", 92):
                raise ValueError("临时图片保存失败")
            return tmp, width, height, ratio
        return path, width, height, 1.0

    def run(self):
        prompt = "%s\n\n%s" % (self.instruction, SCHEMA_TEXT)
        prompt_poly = "%s\n\n%s" % (self.instruction, SCHEMA_TEXT_POLY)
        records, total = [], len(self.images)
        errors = 0

        def infer(idx, path):
            reg = []
            with self._reg_lock:
                self._abort_regs.append(reg)
            try:
                # 每张图片独立新会话：完整指令(标注指令+JSON格式要求+坐标说明)每次全量重发，
                # 且每次请求均为全新 messages（见 api_client._chat），图片之间零上下文共享。
                self.sig_log.emit("[新会话] %s" % os.path.basename(path))
                send_path, width, height, scale = self._prepare_image(path, self.max_side)
                if scale < 1.0:
                    self.sig_log.emit("  ↳ %s 过大，已缩放至最长边 %dpx 再推理，坐标自动换算回原图"
                                      % (os.path.basename(path), int(max(width, height) * scale)))
                boxes, polygons, description = [], [], ""
                if self.mode == "segment" and self.seg_engine == "sam2":
                    # 两阶段：VL 检测框 → SAM2 分割（SAM2 直接读原图，坐标为原图像素）
                    result = chat_json(self.model, prompt, send_path,
                                       self.temperature, self.max_tokens,
                                       log=lambda s: None, abort_registry=reg,
                                       api_config=getattr(self, "api_config", None))
                    description = result["description"]
                    for b in result["boxes"]:
                        boxes.append({
                            "name": b["name"],
                            "x1": round(b["x1"] / scale, 2), "y1": round(b["y1"] / scale, 2),
                            "x2": round(b["x2"] / scale, 2), "y2": round(b["y2"] / scale, 2),
                            "confidence": b["confidence"],
                        })
                    if boxes:
                        self.sig_log.emit("  ↳ %s 分割 %d 个目标…" % (os.path.basename(path), len(boxes)))
                        polygons = self.sam_worker.segment(path, boxes)
                elif self.mode == "segment":
                    # 纯 API 多边形：模型直接输出轮廓点
                    result = chat_polygons(self.model, prompt_poly, send_path,
                                           self.temperature, self.max_tokens,
                                           log=lambda s: None, abort_registry=reg,
                                           api_config=getattr(self, "api_config", None))
                    description = result["description"]
                    for o in result["objects"]:
                        pts = [[p[0] / scale, p[1] / scale] for p in o["points"]]
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        polygons.append(pts)
                        boxes.append({
                            "name": o["name"],
                            "x1": round(min(xs), 2), "y1": round(min(ys), 2),
                            "x2": round(max(xs), 2), "y2": round(max(ys), 2),
                            "confidence": o["confidence"],
                        })
                else:
                    result = chat_json(self.model, prompt, send_path,
                                       self.temperature, self.max_tokens,
                                       log=lambda s: None, abort_registry=reg,
                                       api_config=getattr(self, "api_config", None))
                    description = result["description"]
                    for b in result["boxes"]:
                        boxes.append({
                            "name": b["name"],
                            "x1": round(b["x1"] / scale, 2), "y1": round(b["y1"] / scale, 2),
                            "x2": round(b["x2"] / scale, 2), "y2": round(b["y2"] / scale, 2),
                            "confidence": b["confidence"],
                        })
                rec = {
                    "image_path": os.path.normpath(path),
                    "file_name": os.path.basename(path),
                    "width": width, "height": height,
                    "description": description,
                    "boxes": boxes,
                    "polygons": polygons if len(polygons) == len(boxes) else [],
                    "model": self.model,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "ok", "raw_response": "",
                }
            except ApiCancel:
                raise
            except Exception as e:
                rec = {
                    "image_path": os.path.normpath(path),
                    "file_name": os.path.basename(path),
                    "width": 0, "height": 0,
                    "description": "", "boxes": [], "polygons": [],
                    "model": self.model,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "error", "error": str(e), "raw_response": "",
                }
            finally:
                with self._reg_lock:
                    if reg in self._abort_regs:
                        self._abort_regs.remove(reg)
            return idx, path, rec

        self.sig_log.emit("开始批量标注：共 %d 张图片，模型 %s，并发 %d"
                          % (total, self.model, self.concurrency))
        pending = list(enumerate(self.images))
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {}
            cancelled = False
            while pending or futures:
                while len(futures) < self.concurrency and pending:
                    idx, path = pending.pop(0)
                    futures[pool.submit(infer, idx, path)] = (idx, path)
                if not futures:
                    break
                for fut in as_completed(list(futures), timeout=None):
                    if fut in futures:
                        del futures[fut]
                    try:
                        idx, path, rec = fut.result()
                    except ApiCancel:
                        cancelled = True
                        break
                    records.append(rec)
                    ok = rec["status"] == "ok"
                    if not ok:
                        errors += 1
                        self.sig_log.emit("失败 %s: %s" % (rec["file_name"], rec.get("error")))
                    elif not rec.get("boxes"):
                        self.sig_log.emit("提示 %s: 未检测到目标框（模型描述: %s）"
                                          % (rec["file_name"], (rec.get("description") or "")[:80] or "（无）"))
                    self.sig_item_done.emit({
                        "index": idx, "total": total,
                        "file_name": rec["file_name"],
                        "boxes": len(rec.get("boxes", [])),
                        "status": rec["status"],
                    })
                    break
                if self._stop.is_set() or cancelled:
                    remaining = len(pending) + len(futures)
                    pending.clear()
                    futures.clear()
                    self.sig_log.emit("用户中止，剩余 %d 张未处理。" % remaining)
                    break

        records.sort(key=lambda r: self.images.index(r["image_path"]))

        summary = {"total": total, "ok": total - errors, "failed": errors,
                   "exports": {}, "records": records}
        for fmt, cfg in self.export_config.items():
            if cfg.get("enabled") and cfg.get("dir"):
                try:
                    label, func = EXPORTERS[fmt]
                    n = func(records, cfg["dir"], log=self.sig_log.emit)
                    summary["exports"][fmt] = {"label": label, "dir": cfg["dir"], "files": n}
                    self.sig_log.emit("导出 %s → %s（%d 个文件）" % (label, cfg["dir"], n))
                except Exception as e:
                    self.sig_log.emit("导出 %s 失败: %s" % (fmt, e))
        self.sig_finished.emit(summary)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像自动标注工具")
        fit_to_screen(self, 1180, 760)
        self.settings = QSettings("ollama-annotator", "settings")
        self.runner = None
        self.records = []  # 最近一次批量标注的记录，标注修正在此列表上原地写回

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ------------------------------ 左侧控制面板
        left = QWidget()
        left_layout = QVBoxLayout(left)

        model_box = QGroupBox("连接与模型设置")
        ml = QVBoxLayout(model_box)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Base URL:"))
        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText("http://127.0.0.1:11434/v1")
        row1.addWidget(self.base_url_edit, 1)
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.refresh_models)
        row1.addWidget(self.refresh_btn)
        ml.addLayout(row1)
        row1b = QHBoxLayout()
        row1b.addWidget(QLabel("API Key:"))
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("留空即可（本地服务）")
        row1b.addWidget(self.api_key_edit, 1)
        ml.addLayout(row1b)
        row1c = QHBoxLayout()
        row1c.addWidget(QLabel("模型:"))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setToolTip("点“刷新”从 /v1/models 拉取，也可直接手动输入模型名")
        row1c.addWidget(self.model_combo, 1)
        ml.addLayout(row1c)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("温度:"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.1)
        row2.addWidget(self.temp_spin)
        row2.addSpacing(12)
        row2.addWidget(QLabel("最大token:"))
        self.tokens_spin = QSpinBox()
        self.tokens_spin.setRange(256, 128000)
        self.tokens_spin.setValue(16384)
        self.tokens_spin.setSingleStep(256)
        row2.addWidget(self.tokens_spin)
        row2.addSpacing(12)
        row2.addWidget(QLabel("并发:"))
        self.conc_spin = QSpinBox()
        self.conc_spin.setRange(1, 32)
        self.conc_spin.setValue(4)
        row2.addWidget(self.conc_spin)
        row2.addStretch(1)
        ml.addLayout(row2)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("推理缩放上限(px):"))
        self.max_side_spin = QSpinBox()
        self.max_side_spin.setRange(0, 8192)
        self.max_side_spin.setValue(1600)
        self.max_side_spin.setSingleStep(200)
        self.max_side_spin.setToolTip("图片最长边超过此值将等比缩小后再推理（坐标自动换算回原图）；0=不缩放")
        row3.addWidget(self.max_side_spin)
        row3.addWidget(QLabel("（0=不缩放，超出上下文会报错）"), 1)
        ml.addLayout(row3)
        left_layout.addWidget(model_box)

        seg_box = QGroupBox("标注类型")
        sl = QVBoxLayout(seg_box)
        seg_row = QHBoxLayout()
        seg_row.addWidget(QLabel("标注方式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("目标检测（边界框）", "detect")
        self.mode_combo.addItem("实例分割（轮廓）", "segment")
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        seg_row.addWidget(self.mode_combo, 1)
        sl.addLayout(seg_row)
        eng_row = QHBoxLayout()
        eng_row.addWidget(QLabel("分割引擎:"))
        self.seg_engine_combo = QComboBox()
        self.seg_engine_combo.addItem("API 多边形（模型直接输出轮廓）", "api")
        self.seg_engine_combo.addItem("本地 SAM2（检测框+像素分割）", "sam2")
        eng_row.addWidget(self.seg_engine_combo, 1)
        sl.addLayout(eng_row)
        py_row = QHBoxLayout()
        py_row.addWidget(QLabel("SAM2 环境 Python:"))
        self.sam_python_edit = QLineEdit()
        self.sam_python_edit.setPlaceholderText("自动探测或手动选择（装有 torch+sam2 的 conda 环境 python.exe）")
        self.sam_python_edit.setToolTip("指向装有 torch+sam2 的 conda 环境 python.exe；点“自动查找”可扫描本机 conda 环境")
        py_row.addWidget(self.sam_python_edit, 1)
        auto_btn = QPushButton("自动查找")
        auto_btn.clicked.connect(self.auto_find_sam_python)
        py_row.addWidget(auto_btn)
        py_btn = QPushButton("浏览")
        py_btn.clicked.connect(self.pick_sam_python)
        py_row.addWidget(py_btn)
        sl.addLayout(py_row)
        left_layout.addWidget(seg_box)

        prompt_box = QGroupBox("标注指令（自定义，支持多行）")
        pl = QVBoxLayout(prompt_box)
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("在此编写检测指令，例如：检测图片中的所有汽车，标注品牌与颜色…")
        pl.addWidget(self.prompt_edit)
        btn_row = QHBoxLayout()
        restore_btn = QPushButton("恢复默认指令")
        restore_btn.clicked.connect(lambda: self.prompt_edit.setPlainText(DEFAULT_INSTRUCTION))
        hint = QLabel("※ YOLO/COCO 导出依赖模型返回边界框")
        hint.setStyleSheet("color:gray;")
        btn_row.addWidget(restore_btn)
        btn_row.addWidget(hint, 1)
        pl.addLayout(btn_row)
        left_layout.addWidget(prompt_box)

        export_box = QGroupBox("导出设置（每种格式独立输出目录）")
        gl = QGridLayout(export_box)
        self.export_rows = {}
        for i, fmt in enumerate(EXPORTERS):
            cb = QCheckBox(FORMAT_LABELS[fmt])
            cb.toggled.connect(lambda on, f=fmt: self.export_rows[f][1].setEnabled(on)
                               if on else self.export_rows[f][1].setEnabled(False))
            path_edit = QLineEdit()
            path_edit.setPlaceholderText("未选择，不导出此格式")
            path_edit.setEnabled(False)
            browse_btn = QPushButton("选择文件夹")
            browse_btn.setEnabled(False)
            browse_btn.clicked.connect(lambda _c=False, f=fmt: self.pick_export_dir(f))
            cb.toggled.connect(lambda on, b=browse_btn: b.setEnabled(on))
            self.export_rows[fmt] = (cb, path_edit, browse_btn)
            gl.addWidget(cb, i * 2, 0, 1, 3)
            gl.addWidget(path_edit, i * 2 + 1, 0, 1, 3)
            gl.addWidget(browse_btn, i * 2 + 1, 3)
        left_layout.addWidget(export_box)

        start_row = QHBoxLayout()
        self.start_btn = QPushButton("▶ 开始批量标注")
        self.start_btn.setStyleSheet("QPushButton{font-size:14px;font-weight:bold;padding:8px;}")
        self.start_btn.clicked.connect(self.start_batch)
        self.stop_btn = QPushButton("■ 停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_batch)
        self.clear_btn = QPushButton("清空列表")
        self.clear_btn.clicked.connect(lambda: self.image_list.clear())
        start_row.addWidget(self.start_btn, 2)
        start_row.addWidget(self.stop_btn, 1)
        start_row.addWidget(self.clear_btn, 1)
        left_layout.addLayout(start_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress_label = QLabel("就绪。将图片或文件夹拖入右侧列表。")
        left_layout.addWidget(self.progress)
        left_layout.addWidget(self.progress_label)

        left_layout.addWidget(QLabel("运行日志:"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(160)
        left_layout.addWidget(self.log_view, 1)

        splitter.addWidget(left)

        # ------------------------------ 右侧图片列表
        right = QFrame()
        rl = QVBoxLayout(right)
        tip = QLabel("🎯 批量图片列表 —— 将图片文件或文件夹直接拖入此处（可多选 / 多次拖入）")
        tip.setStyleSheet("color:#666;")
        rl.addWidget(tip)
        self.image_list = ImageListWidget()
        self.image_list.dropped.connect(self.on_dropped)
        self.image_list.itemDoubleClicked.connect(self.open_editor_for)
        rl.addWidget(self.image_list, 1)
        count_row = QHBoxLayout()
        self.count_label = QLabel("共 0 张图片")
        add_btn = QPushButton("添加图片…")
        add_btn.clicked.connect(self.add_images_dialog)
        add_btn2 = QPushButton("添加文件夹…")
        add_btn2.clicked.connect(self.add_folder_dialog)
        count_row.addWidget(self.count_label)
        count_row.addStretch(1)
        count_row.addWidget(add_btn)
        count_row.addWidget(add_btn2)
        rl.addLayout(count_row)
        edit_row = QHBoxLayout()
        fix_btn = QPushButton("✎ 修正标注")
        fix_btn.setToolTip("打开修正窗口，手动拖拽/增删轮廓点（也可双击图片列表项）")
        fix_btn.clicked.connect(lambda: self.open_editor(None))
        export_fixed_btn = QPushButton("导出修正后结果")
        export_fixed_btn.setToolTip("用修正后的标注数据重新执行已勾选的导出格式")
        export_fixed_btn.clicked.connect(self.export_corrected)
        edit_row.addWidget(fix_btn)
        edit_row.addWidget(export_fixed_btn)
        rl.addLayout(edit_row)
        splitter.addWidget(right)

        splitter.setSizes([520, 660])

        self._load_settings()

    # ------------------------------------------------------- 通用
    def log(self, text):
        self.log_view.appendPlainText(text)

    def on_dropped(self, paths):
        images = collect_images(paths)
        if not images:
            self.log("拖入内容中没有图片文件。")
            return
        for img in images:
            if not self._path_in_list(img):
                item = QListWidgetItem(os.path.basename(img))
                item.setToolTip(img)
                item.setData(Qt.ItemDataRole.UserRole, img)
                icon = QIcon(img)
                if not icon.isNull():
                    item.setIcon(icon)
                self.image_list.addItem(item)
        self.log("已添加 %d 张图片（当前共 %d 张）。" % (len(images), self.image_list.count()))
        self.update_count()

    def _path_in_list(self, path):
        for i in range(self.image_list.count()):
            if self.image_list.item(i).data(Qt.ItemDataRole.UserRole) == path:
                return True
        return False

    def add_images_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "",
            "图片 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.gif)")
        if files:
            self.on_dropped(files)

    def add_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            self.on_dropped([folder])

    def update_count(self):
        self.count_label.setText("共 %d 张图片" % self.image_list.count())

    def pick_export_dir(self, fmt):
        folder = QFileDialog.getExistingDirectory(self, "选择 %s 导出目录" % FORMAT_LABELS[fmt])
        if folder:
            self.export_rows[fmt][1].setText(folder)

    # ------------------------------------------------------- 连接与模型
    def on_mode_changed(self, _idx):
        seg_mode = self.mode_combo.currentData() == "segment"
        self.seg_engine_combo.setEnabled(seg_mode)
        self.sam_python_edit.setEnabled(seg_mode and self.seg_engine_combo.currentData() == "sam2")

    def pick_sam_python(self):
        start = os.path.expanduser("~")
        cur = self.sam_python_edit.text().strip()
        if cur and os.path.isdir(os.path.dirname(cur)):
            start = os.path.dirname(cur)
        path, _ = QFileDialog.getOpenFileName(self, "选择 SAM2 环境 Python", start, "python.exe")
        if path:
            self.sam_python_edit.setText(path)

    def auto_find_sam_python(self):
        """扫描本机 conda 环境，自动填充装有 sam2 的 Python 路径。"""
        self.log("正在自动探测 SAM2 环境（扫描 conda 环境目录）…")
        found = find_sam2_python()
        if found:
            self.sam_python_edit.setText(found)
            self.log("已找到 SAM2 环境：%s" % found)
            QMessageBox.information(self, "自动查找", "已找到 SAM2 环境：\n%s" % found)
        else:
            self.log("未找到装有 sam2 的 conda 环境。")
            QMessageBox.information(self, "自动查找",
                                    "未在本机找到装有 sam2 的 conda 环境。\n\n"
                                    "请先按 README“实例分割：SAM2 本地引擎”章节创建环境并安装，"
                                    "或改用“API 多边形”分割引擎（无需本地环境）。")

    def _ensure_sam_worker(self, python_path):
        """按需创建/复用 SAM2 常驻子进程（换路径时重建）。"""
        cur = getattr(self, "sam_worker", None)
        cur_path = getattr(cur, "python", None)
        if cur is not None and cur_path == python_path and cur.proc is not None and cur.proc.poll() is None:
            return cur
        if cur is not None:
            cur.close()
        self.sam_worker = SamWorker(python_path, log=self.log)
        return self.sam_worker

    def _get_sam_worker(self):
        """获取可用的 SAM2 worker；环境未配置时自动探测，不可用返回 None。"""
        python_path = self.sam_python_edit.text().strip()
        if not python_path or not os.path.isfile(python_path):
            found = find_sam2_python()
            if found:
                python_path = found
                self.sam_python_edit.setText(found)
        if not python_path or not os.path.isfile(python_path):
            return None
        try:
            return self._ensure_sam_worker(python_path)
        except Exception as e:
            self.log("SAM2 启动失败: %s" % e)
            return None

    def sam_segment_points(self, image_path, points, labels):
        """供编辑窗口调用的点提示分割；返回 polygon 或 None；不可用时抛 ApiError。
        该方法可能在后台线程被调用，故不弹窗，只抛异常。"""
        worker = self._get_sam_worker()
        if worker is None:
            raise ApiError("未找到装有 torch+sam2 的 conda 环境，请先在主界面配置 SAM2 环境。")
        polys = worker.segment_points(image_path, points, labels)
        return polys[0] if polys else None

    def api_config(self):
        return {
            "base_url": self.base_url_edit.text().strip(),
            "api_key": self.api_key_edit.text().strip(),
        }

    def refresh_models(self):
        api = self.api_config()
        if not api["base_url"]:
            QMessageBox.warning(self, "提示", "请先填写 Base URL（如 http://127.0.0.1:11434/v1）。")
            return
        try:
            models = list_vision_models(api)
        except ApiError as e:
            self.log("获取模型失败：%s" % e)
            QMessageBox.warning(self, "API 服务未连接",
                                "无法连接 %s（%s）。\n"
                                "请确认服务地址与 API Key 正确，"
                                "模型名也可以直接在输入框中手动填写。" % (api["base_url"], e))
            return
        if not models:
            self.log("未获取到模型列表，可在模型输入框手动填写模型名。")
            self.model_combo.clear()
            return
        saved = self.model_combo.currentText()
        self.model_combo.clear()
        for m in models:
            self.model_combo.addItem(m["name"])
        if saved:
            idx = self.model_combo.findText(saved)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        self.log("已从 %s 获取 %d 个模型。" % (api["base_url"], len(models)))

    def current_model_name(self):
        """返回当前选中的模型名（兼容手动输入）。"""
        name = self.model_combo.currentData() or self.model_combo.currentText().strip()
        return name

    # ------------------------------------------------------- 批量任务
    def start_batch(self):
        if self.runner and self.runner.isRunning():
            return
        images = [self.image_list.item(i).data(Qt.ItemDataRole.UserRole)
                  for i in range(self.image_list.count())]
        if not images:
            QMessageBox.information(self, "提示", "请先拖入图片。")
            return
        if not self.base_url_edit.text().strip():
            QMessageBox.warning(self, "提示", "请先填写 Base URL（如 http://127.0.0.1:11434/v1）。")
            return
        model_name = self.current_model_name()
        if not model_name:
            QMessageBox.warning(self, "提示", "没有可用视觉模型，请刷新获取或手动输入模型名。")
            return
        instruction = self.prompt_edit.toPlainText().strip()
        if not instruction:
            QMessageBox.warning(self, "提示", "请填写标注指令。")
            return

        export_config = {}
        for fmt, (cb, path_edit, _btn) in self.export_rows.items():
            export_config[fmt] = {
                "enabled": cb.isChecked(),
                "dir": path_edit.text().strip(),
            }
        if not any(c["enabled"] and c["dir"] for c in export_config.values()):
            QMessageBox.information(self, "提示", "未勾选任何导出格式，将只显示标注结果不落盘。")
            self.log("警告：未选择任何导出格式。")

        mode = self.mode_combo.currentData()
        seg_engine = self.seg_engine_combo.currentData() if mode == "segment" else None
        sam_worker = None
        if mode == "segment" and seg_engine == "sam2":
            python_path = self.sam_python_edit.text().strip()
            if not python_path or not os.path.isfile(python_path):
                self.log("SAM2 环境路径无效，尝试自动探测…")
                found = find_sam2_python()
                if found:
                    python_path = found
                    self.sam_python_edit.setText(found)
                    self.log("已自动探测到 SAM2 环境：%s" % found)
            if not python_path or not os.path.isfile(python_path):
                QMessageBox.warning(self, "提示", "找不到 SAM2 环境 Python：\n%s\n\n"
                                    "请点右侧“自动查找”，或选择装有 torch+sam2 的 conda 环境 python.exe。" % python_path)
                return
            ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "checkpoints", "sam2.1_hiera_tiny.pt")
            if not os.path.isfile(ckpt):
                QMessageBox.warning(self, "缺少 SAM2 权重",
                                    "未找到 SAM2 权重文件：\n%s\n\n"
                                    "请按 README“实例分割：SAM2 本地引擎”章节下载权重到 checkpoints/ 目录。" % ckpt)
                return
            try:
                sam_worker = self._ensure_sam_worker(python_path)
            except Exception as e:
                QMessageBox.warning(self, "SAM2 启动失败",
                                    "无法启动 SAM2 分割进程（%s）。\n"
                                    "请确认该环境已安装 torch 与 sam2（pip install sam2）。" % e)
                return

        self._save_settings()
        self.progress.setRange(0, len(images))
        self.progress.setValue(0)
        self.progress_label.setText("0/%d 开始标注…" % len(images))
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.runner = BatchRunner(
            images,
            model_name,
            instruction,
            self.temp_spin.value(),
            self.tokens_spin.value(),
            self.conc_spin.value(),
            export_config,
            self.max_side_spin.value(),
            self,
            mode=mode, seg_engine=seg_engine, sam_worker=sam_worker,
        )
        self.runner.api_config = self.api_config()
        self.runner.sig_item_done.connect(self.on_item_done)
        self.runner.sig_finished.connect(self.on_finished)
        self.runner.sig_log.connect(self.log)
        self.runner.start()

    def stop_batch(self):
        if self.runner and self.runner.isRunning():
            self.runner.stop()
            self.log("正在停止…")

    def on_item_done(self, info):
        self.progress.setValue(info["index"] + 1)
        self.progress_label.setText("%d/%d 完成：%s（框数 %d）" % (
            info["index"] + 1, info["total"], info["file_name"], info["boxes"]))

    def on_finished(self, summary):
        self.progress.setValue(summary["total"])
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_label.setText("完成：成功 %d / 失败 %d / 共 %d" % (
            summary["ok"], summary["failed"], summary["total"]))
        if summary["failed"]:
            self.log("标注完成：成功 %d，失败 %d（失败图片仍会导出空标注）。"
                     % (summary["ok"], summary["failed"]))
        else:
            self.log("全部 %d 张标注成功。" % summary["total"])
        if summary["exports"]:
            self.log("— 导出汇总 —")
            for fmt, info in summary["exports"].items():
                self.log("  %s: %s（%d 个文件）" % (info["label"], info["dir"], info["files"]))
        self.records[:] = summary.get("records", [])
        self.log("已保存 %d 条标注记录，可在右侧列表双击图片打开修正窗口，或点“导出修正后结果”。"
                 % len(self.records))
        msg = "批量标注完成\n成功 %d / 失败 %d" % (summary["ok"], summary["failed"])
        if summary["exports"]:
            msg += "\n\n导出目录：\n" + "\n".join(
                "• %s → %s" % (i["label"], i["dir"]) for i in summary["exports"].values())
        QMessageBox.information(self, "完成", msg)

    # ------------------------------------------------------- 交互式修正
    def _rec_for_item(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return None
        path = os.path.normpath(path)
        for r in self.records:
            if os.path.normpath(r.get("image_path", "")) == path:
                return r
        return None

    def open_editor_for(self, item):
        self.open_editor(self._rec_for_item(item))

    def open_editor(self, rec=None):
        ok_recs = [r for r in self.records if r.get("status") == "ok"]
        if not ok_recs:
            QMessageBox.information(self, "提示", "请先运行批量标注，再进行标注修正。")
            return
        index = 0
        if rec is not None and rec in ok_recs:
            index = ok_recs.index(rec)
        win = AnnotationEditorWindow(self.records, index, self,
                                     sam_segmenter=self.sam_segment_points)
        win.show()
        win.raise_()
        win.activateWindow()

    def export_corrected(self):
        if not self.records:
            QMessageBox.information(self, "提示", "请先运行批量标注，再导出修正结果。")
            return
        enabled = {}
        for fmt, (cb, path_edit, _btn) in self.export_rows.items():
            if cb.isChecked() and path_edit.text().strip():
                enabled[fmt] = path_edit.text().strip()
        if not enabled:
            QMessageBox.information(self, "提示", "未勾选任何导出格式或未选择导出目录。")
            return
        self.log("— 重新导出（修正后数据）—")
        for fmt, out_dir in enabled.items():
            try:
                label, func = EXPORTERS[fmt]
                n = func(self.records, out_dir, log=self.log)
                self.log("导出 %s → %s（%d 个文件）" % (label, out_dir, n))
            except Exception as e:
                self.log("导出 %s 失败: %s" % (fmt, e))
        QMessageBox.information(self, "完成", "修正后数据已重新导出到已勾选格式的目录。")

    # ------------------------------------------------------- 设置持久化
    def _save_settings(self):
        s = self.settings
        s.setValue("base_url", self.base_url_edit.text().strip())
        s.setValue("api_key", self.api_key_edit.text())
        s.setValue("model", self.model_combo.currentText())
        s.setValue("instruction", self.prompt_edit.toPlainText())
        s.setValue("temperature", self.temp_spin.value())
        s.setValue("max_tokens", self.tokens_spin.value())
        s.setValue("concurrency", self.conc_spin.value())
        s.setValue("max_side", self.max_side_spin.value())
        s.setValue("mode", self.mode_combo.currentData())
        s.setValue("seg_engine", self.seg_engine_combo.currentData())
        s.setValue("sam_python", self.sam_python_edit.text().strip())
        for fmt, (cb, path_edit, _b) in self.export_rows.items():
            s.setValue("export_%s_on" % fmt, cb.isChecked())
            s.setValue("export_%s_dir" % fmt, path_edit.text().strip())

    def _load_settings(self):
        s = self.settings
        base_url = s.value("base_url", "")
        if base_url:
            self.base_url_edit.setText(base_url)
        self.api_key_edit.setText(s.value("api_key", ""))
        mode = s.value("mode", "detect")
        idx = self.mode_combo.findData(mode)
        if idx >= 0:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(idx)
            self.mode_combo.blockSignals(False)
        seg_engine = s.value("seg_engine", "sam2")
        idx = self.seg_engine_combo.findData(seg_engine)
        if idx >= 0:
            self.seg_engine_combo.setCurrentIndex(idx)
        sam_python = s.value("sam_python", "")
        if sam_python:
            self.sam_python_edit.setText(sam_python)
        self.on_mode_changed(None)
        self.prompt_edit.setPlainText(
            s.value("instruction", DEFAULT_INSTRUCTION))
        self.temp_spin.setValue(float(s.value("temperature", 0.1)))
        self.tokens_spin.setValue(int(s.value("max_tokens", 16384)))
        self.conc_spin.setValue(int(s.value("concurrency", 4)))
        self.max_side_spin.setValue(int(s.value("max_side", 1600)))
        saved_model = s.value("model", "")
        if saved_model:
            self.model_combo.addItem(saved_model)
            self.model_combo.setCurrentText(saved_model)
        for fmt, (cb, path_edit, _b) in self.export_rows.items():
            on = s.value("export_%s_on" % fmt, "false") == "true"
            d = s.value("export_%s_dir" % fmt, "")
            cb.setChecked(on)
            path_edit.setText(d if d else "")
            path_edit.setEnabled(on)
            self.export_rows[fmt][2].setEnabled(on)

    def closeEvent(self, event):
        worker = getattr(self, "sam_worker", None)
        if worker is not None:
            worker.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()