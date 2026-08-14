# -*- coding: utf-8 -*-
"""SAM2 实例分割子进程服务（运行于装有 torch+sam2 的 conda 环境）。

主程序（无 torch 的 base 环境）通过子进程调用本脚本，协议为 JSON 行：

请求 (stdin, 每行一个任务):
    {"image": "D:/a.jpg", "boxes": [{"x1": 0, "y1": 0, "x2": 100, "y2": 100}, ...]}

响应 (stdout, 每行一个结果，与请求顺序一一对应):
    {"polygons": [[[x, y], ...], ...], "error": null}
    # polygons[i] 为第 i 个框的轮廓点列表（原图坐标，已简化）；
    # 一个框可能对应多个连通域，取最大连通域；失败时 polygons[i] 为 []

用法:
    python sam_cli.py --checkpoint ckpt.pt [--config cfg.yaml] [--serve]
    --serve: 常驻模式（默认）
    --once --image xx --boxes '[...]': 单次模式（测试用）
"""
import argparse
import json
import sys
import time

CKPT_DEFAULT = r"D:\葡萄\物体识别标注\checkpoints\sam2.1_hiera_tiny.pt"
CONFIG_DEFAULT = "configs/sam2.1/sam2.1_hiera_t.yaml"


def load_model(checkpoint, config):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(config, checkpoint, device="cuda")
    return SAM2ImagePredictor(model)


def mask_to_polygon(mask, epsilon=2.0):
    """掩码转多边形（最大连通域，近似简化）。mask: HxW bool。返回 [[x, y], ...] 或 []。"""
    import cv2
    import numpy as np
    m = mask.astype(np.uint8)
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


class Segmenter:
    def __init__(self, checkpoint, config):
        self.predictor = load_model(checkpoint, config)
        self._image = None

    def segment(self, image_path, boxes):
        """对每框中心点做提示分割，返回多边形列表。"""
        import cv2
        import numpy as np
        try:
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            return [], "无法读取图片: %s" % image_path
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        polys = []
        for b in boxes:
            try:
                cx = (float(b["x1"]) + float(b["x2"])) / 2
                cy = (float(b["y1"]) + float(b["y2"])) / 2
                masks, scores, _ = self.predictor.predict(
                    point_coords=np.array([[cx, cy]], dtype=np.float32),
                    point_labels=np.array([1]),
                    multimask_output=True,
                )
                best = int(scores.argmax())
                polys.append(mask_to_polygon(masks[best]))
            except Exception as e:
                polys.append([])
        return polys, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT_DEFAULT)
    ap.add_argument("--config", default=CONFIG_DEFAULT)
    ap.add_argument("--serve", action="store_true", help="常驻模式（默认）")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--image")
    ap.add_argument("--boxes", help="JSON 数组字符串")
    args = ap.parse_args()

    t0 = time.time()
    seg = Segmenter(args.checkpoint, args.config)
    sys.stderr.write("[sam_cli] 模型加载完成 %.1fs\n" % (time.time() - t0))
    sys.stderr.flush()

    def handle(task):
        try:
            image = task["image"]
            boxes = task.get("boxes") or []
            polys, err = seg.segment(image, boxes)
            return {"polygons": polys, "error": err}
        except Exception as e:
            return {"polygons": [], "error": "%s: %s" % (type(e).__name__, e)}

    if args.once:
        task = {"image": args.image, "boxes": json.loads(args.boxes)}
        print(json.dumps(handle(task), ensure_ascii=False))
        return

    # 常驻服务：逐行读取请求，逐行输出结果
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            task = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"polygons": [], "error": "bad request"}), flush=True)
            continue
        print(json.dumps(handle(task), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()