# -*- coding: utf-8 -*-
"""JSON / YOLO TXT / COCO / Markdown 四种标注格式导出器。

records 中每条记录的字段:
    image_path, file_name, width, height,
    description, boxes: [{name,x1,y1,x2,y2,confidence}],
    model, timestamp, status("ok"/"error"), raw_response
"""
import json
import os
import shutil

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


def collect_images(paths):
    """收集拖入的文件/文件夹中的所有图片，返回去重后的路径列表。"""
    images, seen = [], set()
    for p in paths:
        p = os.path.normpath(p)
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in files:
                    if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                        full = os.path.join(root, f)
                        if full not in seen:
                            seen.add(full)
                            images.append(full)
        elif os.path.isfile(p) and os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS:
            if p not in seen:
                seen.add(p)
                images.append(p)
    return images


def _mkdir(out_dir):
    os.makedirs(out_dir, exist_ok=True)


# ---------------------------------------------------------------- JSON
def export_json(records, out_dir, log=None):
    _mkdir(out_dir)
    count = 0
    for rec in records:
        target = os.path.join(out_dir, os.path.splitext(rec["file_name"])[0] + ".json")
        with open(target, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        count += 1
    return count


# ---------------------------------------------------------------- YOLO TXT
def _yolo_classes(records):
    """按首次出现顺序收集类别名。"""
    classes, seen = [], set()
    for rec in records:
        for b in rec.get("boxes", []):
            name = b["name"]
            if name not in seen:
                seen.add(name)
                classes.append(name)
    return classes


def _rec_polygons(rec):
    """返回与 boxes 一一对应的多边形列表（无则为 []）。"""
    polys = rec.get("polygons") or []
    return polys if len(polys) == len(rec.get("boxes", [])) else []


def _yolo_seg_line(cls_idx, b, poly, w, h):
    """YOLO-seg 行：class cx cy w h x1 y1 x2 y2 ...（多边形点归一化）。"""
    def clamp(v):
        return min(max(v, 0.0), 1.0)
    cx = ((b["x1"] + b["x2"]) / 2.0) / max(w, 1)
    cy = ((b["y1"] + b["y2"]) / 2.0) / max(h, 1)
    bw = (b["x2"] - b["x1"]) / max(w, 1)
    bh = (b["y2"] - b["y1"]) / max(h, 1)
    nums = ["%d" % cls_idx, "%.6f" % clamp(cx), "%.6f" % clamp(cy),
            "%.6f" % clamp(bw), "%.6f" % clamp(bh)]
    if poly:
        nums += ["%.6f" % clamp(px / max(w, 1)) for px, _py in poly]
        nums += ["%.6f" % clamp(py / max(h, 1)) for _px, py in poly]
    return " ".join(nums)


def export_yolo(records, out_dir, log=None):
    _mkdir(out_dir)
    classes = _yolo_classes(records)
    idx = {name: i for i, name in enumerate(classes)}
    seg_used = False
    count = 0
    for rec in records:
        w, h = rec["width"], rec["height"]
        lines = []
        polys = _rec_polygons(rec)
        for i, b in enumerate(rec.get("boxes", [])):
            poly = polys[i] if polys else []
            if poly:
                seg_used = True
            lines.append(_yolo_seg_line(idx[b["name"]], b, poly, w, h))
        target = os.path.join(out_dir, os.path.splitext(rec["file_name"])[0] + ".txt")
        with open(target, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        count += 1
    with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(classes))
    if log and seg_used:
        log("YOLO 已按分割格式导出（class cx cy w h + 多边形点）")
    return count


# ---------------------------------------------------------------- COCO
def export_coco(records, out_dir, log=None):
    _mkdir(out_dir)
    classes = _yolo_classes(records)
    cat_id = {name: i + 1 for i, name in enumerate(classes)}

    coco = {"images": [], "annotations": [], "categories": []}
    for i, name in enumerate(classes):
        coco["categories"].append({"id": i + 1, "name": name, "supercategory": "object"})

    ann_id = 1
    for img_id, rec in enumerate(records, start=1):
        coco["images"].append({
            "id": img_id,
            "file_name": rec["file_name"],
            "width": rec["width"],
            "height": rec["height"],
        })
        polys = _rec_polygons(rec)
        for i, b in enumerate(rec.get("boxes", [])):
            x, y = b["x1"], b["y1"]
            w, h = b["x2"] - b["x1"], b["y2"] - b["y1"]
            poly = polys[i] if polys else []
            if poly:
                seg = [round(float(p), 2) for pt in poly for p in pt]
            else:
                seg = [round(x, 2), round(y, 2), round(x + w, 2), round(y, 2),
                       round(x + w, 2), round(y + h, 2), round(x, 2), round(y + h, 2)]
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id[b["name"]],
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "iscrowd": 0,
                "score": b["confidence"],
                "segmentation": [seg],
            })
            ann_id += 1

    target = os.path.join(out_dir, "annotations.json")
    with open(target, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    return 1


# ---------------------------------------------------------------- Markdown
def export_markdown(records, out_dir, log=None):
    _mkdir(out_dir)
    count = 0
    for rec in records:
        stem = os.path.splitext(rec["file_name"])[0]
        md_path = os.path.join(out_dir, stem + ".md")
        img_link = rec["file_name"]
        try:
            dest = os.path.join(out_dir, rec["file_name"])
            if not os.path.exists(dest):
                shutil.copy2(rec["image_path"], dest)
        except OSError:
            img_link = rec["image_path"]

        rows = []
        polys = _rec_polygons(rec)
        for i, b in enumerate(rec.get("boxes", []), start=1):
            np_pts = len(polys[i - 1]) if polys else 0
            rows.append("| %d | %s | %.2f | %s | %s | %s | %s | %s |" % (
                i, b["name"], b["confidence"],
                int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"]),
                np_pts if np_pts else "-"))

        desc = rec.get("description") or "_（无描述）_"
        if rec["status"] != "ok":
            desc = "> ⚠️ 本图标注失败：%s\n\n%s" % (rec.get("error", ""), desc)

        md = [
            "# %s" % rec["file_name"],
            "",
            "![%s](%s)" % (rec["file_name"], img_link),
            "",
            "## 图片描述",
            "",
            desc,
            "",
            "## 目标标注",
            "",
            "| # | 类别 | 置信度 | x1 | y1 | x2 | y2 | 轮廓点 |",
            "|---|------|--------|----|----|----|----|--------|",
        ] + rows
        if not rows:
            md += ["", "_（未检测到目标）_"]
        md += ["", "---", "",
               "> 模型: `%s` · 生成时间: %s" % (rec["model"], rec["timestamp"])]
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
        count += 1
    return count


EXPORTERS = {
    "json": ("JSON", export_json),
    "yolo": ("YOLO TXT", export_yolo),
    "coco": ("COCO", export_coco),
    "markdown": ("Markdown", export_markdown),
}