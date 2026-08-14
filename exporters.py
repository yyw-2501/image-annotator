# -*- coding: utf-8 -*-
"""JSON / YOLO TXT / COCO / Markdown 四种标注格式导出器。

records 中每条记录的字段:
    image_path, file_name, width, height,
    description, boxes: [{name,x1,y1,x2,y2,confidence}],
    model, timestamp, status("ok"/"error"), raw_response
"""
import csv
import json
import os
import shutil
import xml.etree.ElementTree as ET

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


# ---------------------------------------------------------------- Pascal VOC XML
def export_voc(records, out_dir, log=None):
    """Pascal VOC 格式：每图一个 .xml，含 bndbox；有分割数据时附 polygon 扩展标签。"""
    _mkdir(out_dir)
    count = 0
    for rec in records:
        ann = ET.Element("annotation")
        ET.SubElement(ann, "folder").text = os.path.basename(out_dir)
        ET.SubElement(ann, "filename").text = rec["file_name"]
        src = ET.SubElement(ann, "source")
        ET.SubElement(src, "database").text = "image-annotator"
        size = ET.SubElement(ann, "size")
        ET.SubElement(size, "width").text = str(rec["width"])
        ET.SubElement(size, "height").text = str(rec["height"])
        ET.SubElement(size, "depth").text = "3"
        ET.SubElement(ann, "segmented").text = "1"
        polys = _rec_polygons(rec)
        for i, b in enumerate(rec.get("boxes", [])):
            obj = ET.SubElement(ann, "object")
            ET.SubElement(obj, "name").text = b["name"]
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            ET.SubElement(obj, "confidence").text = "%.4f" % b["confidence"]
            bb = ET.SubElement(obj, "bndbox")
            ET.SubElement(bb, "xmin").text = str(int(b["x1"]))
            ET.SubElement(bb, "ymin").text = str(int(b["y1"]))
            ET.SubElement(bb, "xmax").text = str(int(b["x2"]))
            ET.SubElement(bb, "ymax").text = str(int(b["y2"]))
            if polys and polys[i]:
                pg = ET.SubElement(obj, "polygon")
                for x, y in polys[i]:
                    pt = ET.SubElement(pg, "pt")
                    ET.SubElement(pt, "x").text = str(int(round(x)))
                    ET.SubElement(pt, "y").text = str(int(round(y)))
        target = os.path.join(out_dir, os.path.splitext(rec["file_name"])[0] + ".xml")
        ET.ElementTree(ann).write(target, encoding="utf-8", xml_declaration=True)
        count += 1
    return count


# ---------------------------------------------------------------- LabelMe
def export_labelme(records, out_dir, log=None):
    """LabelMe 格式：每图一个 .json，shapes 为多边形（无分割时用 rectangle）。"""
    _mkdir(out_dir)
    count = 0
    for rec in records:
        shapes = []
        polys = _rec_polygons(rec)
        for i, b in enumerate(rec.get("boxes", [])):
            if polys and polys[i]:
                shapes.append({
                    "label": b["name"],
                    "points": [[float(x), float(y)] for x, y in polys[i]],
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {},
                })
            else:
                shapes.append({
                    "label": b["name"],
                    "points": [[float(b["x1"]), float(b["y1"])],
                               [float(b["x2"]), float(b["y2"])]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {},
                })
        data = {
            "version": "5.2.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": rec["file_name"],
            "imageData": None,
            "imageWidth": rec["width"],
            "imageHeight": rec["height"],
            "description": rec.get("description", ""),
        }
        target = os.path.join(out_dir, os.path.splitext(rec["file_name"])[0] + ".json")
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        count += 1
    return count


# ---------------------------------------------------------------- CSV 汇总
def export_csv(records, out_dir, log=None):
    """汇总 CSV：每目标一行（图片、类别、坐标、置信度、轮廓点数），另附图片描述列。"""
    _mkdir(out_dir)
    target = os.path.join(out_dir, "annotations.csv")
    desc_map = {rec["file_name"]: rec.get("description", "") for rec in records}
    with open(target, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class", "confidence", "x1", "y1", "x2", "y2",
                         "width", "height", "polygon_points", "description"])
        for rec in records:
            polys = _rec_polygons(rec)
            for i, b in enumerate(rec.get("boxes", [])):
                writer.writerow([
                    rec["file_name"], b["name"], b["confidence"],
                    int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"]),
                    rec["width"], rec["height"],
                    len(polys[i]) if polys and polys[i] else 0,
                    desc_map.get(rec["file_name"], ""),
                ])
    return 1


# ---------------------------------------------------------------- 语义分割掩码 PNG
def export_mask(records, out_dir, log=None):
    """语义分割掩码：每图一个 PNG，像素值=类别 id（背景 0）。有轮廓用多边形填充，否则用 bbox 填充。"""
    from PIL import Image, ImageDraw
    _mkdir(out_dir)
    classes = _yolo_classes(records)
    cls_id = {name: i + 1 for i, name in enumerate(classes)}
    count = 0
    for rec in records:
        w, h = rec["width"], rec["height"]
        if w <= 0 or h <= 0:
            continue
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        polys = _rec_polygons(rec)
        for i, b in enumerate(rec.get("boxes", [])):
            cid = cls_id[b["name"]]
            if polys and polys[i]:
                pts = [(int(round(x)), int(round(y))) for x, y in polys[i]]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=cid)
                    continue
            draw.rectangle([int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])], fill=cid)
        target = os.path.join(out_dir, os.path.splitext(rec["file_name"])[0] + "_mask.png")
        mask.save(target, "PNG")
        count += 1
    if log and classes:
        log("掩码类别索引: %s" % ", ".join("%s=%d" % (n, cls_id[n]) for n in classes))
    return count


EXPORTERS = {
    "json": ("JSON", export_json),
    "yolo": ("YOLO TXT", export_yolo),
    "coco": ("COCO", export_coco),
    "markdown": ("Markdown", export_markdown),
    "voc": ("Pascal VOC XML", export_voc),
    "labelme": ("LabelMe JSON", export_labelme),
    "csv": ("CSV 汇总", export_csv),
    "mask": ("分割掩码 PNG", export_mask),
}