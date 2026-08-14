# -*- coding: utf-8 -*-
"""本地模型推理子进程服务（ultralytics YOLO 权重：检测 / 实例分割）。

运行于装有 torch + ultralytics 的 conda 环境。协议为 JSON 行（stdin/stdout）：

请求 (stdin, 每行一个任务):
    {"image": "D:/a.jpg", "conf": 0.25}

响应 (stdout, 每行一个结果，与请求顺序一一对应):
    {"objects": [{"name": "类别", "x1": 0, "y1": 0, "x2": 100, "y2": 100,
                  "confidence": 0.9, "points": [[x, y], ...]}], "error": null}
    # points 为轮廓多边形（原图像素坐标）；检测权重时为空列表

用法:
    python local_cli.py --weights xx.pt [--serve]      # 常驻模式（默认）
    python local_cli.py --weights xx.pt --once --image a.jpg [--conf 0.25]
"""
import argparse
import json
import sys


def _mask_to_polygon(mask_np, epsilon=2.0):
    """二值掩码转多边形（最大连通域，近似简化）。mask_np: HxW 数组。"""
    import cv2
    import numpy as np
    m = np.asarray(mask_np)
    if m.ndim == 3:
        m = m[0]
    m = (m > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 8:
        return []
    poly = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
    if len(poly) < 3:
        return []
    return [[float(x), float(y)] for x, y in poly]


def _to_float(v):
    try:
        return float(v.item())
    except Exception:
        return float(v)


class Detector:
    def __init__(self, weights):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.weights = weights
        self.task = getattr(self.model, "task", "detect")

    def infer(self, image_path, conf=0.25):
        results = self.model.predict(source=image_path, conf=conf, verbose=False)
        objects = []
        for r in results:
            names = r.names or {}
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            masks = getattr(r, "masks", None)
            mask_data = masks.data if masks is not None else None
            for i in range(len(boxes)):
                b = boxes[i]
                xyxy = b.xyxy[0].tolist()
                cls = int(_to_float(b.cls[0]))
                obj = {
                    "name": str(names.get(cls, str(cls))),
                    "x1": round(float(xyxy[0]), 2),
                    "y1": round(float(xyxy[1]), 2),
                    "x2": round(float(xyxy[2]), 2),
                    "y2": round(float(xyxy[3]), 2),
                    "confidence": round(_to_float(b.conf[0]), 4),
                    "points": [],
                }
                if mask_data is not None and i < len(mask_data):
                    import numpy as np
                    mn = mask_data[i].cpu().numpy() if hasattr(mask_data[i], "cpu") else np.asarray(mask_data[i])
                    obj["points"] = _mask_to_polygon(mn)
                objects.append(obj)
        return objects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--serve", action="store_true", help="常驻模式（默认）")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--image")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    det = Detector(args.weights)
    sys.stderr.write("[local_cli] 权重加载完成（task=%s）: %s\n" % (det.task, args.weights))
    sys.stderr.flush()

    def handle(task):
        try:
            objs = det.infer(task["image"], task.get("conf", 0.25))
            return {"objects": objs, "error": None}
        except Exception as e:
            return {"objects": [], "error": "%s: %s" % (type(e).__name__, e)}

    if args.once:
        print(json.dumps(handle({"image": args.image, "conf": args.conf}), ensure_ascii=False))
        return

    # 常驻服务：逐行读取请求，逐行输出结果
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            task = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"objects": [], "error": "bad request"}), flush=True)
            continue
        print(json.dumps(handle(task), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
