# -*- coding: utf-8 -*-
"""视觉模型客户端封装：OpenAI 兼容 API（可接 LM Studio、vLLM、Ollama /v1、云端 API 等）。

api_config:
    {"base_url": "http://127.0.0.1:1234/v1", "api_key": ""}
"""
import base64
import http.client
import json
import os
import re
import socket
import ssl
import urllib.parse

SCHEMA_TEXT = (
    '输出 JSON 格式（必须严格符合，且不得输出 JSON 之外的任何内容）：\n'
    '{"description": "一句话图片描述", "boxes": [{"name": "类别名", "x1": 0, "y1": 0, "x2": 0, "y2": 0, "confidence": 0.95}]}\n'
    "坐标说明：均为像素坐标，原点(0,0)在图片左上角，x向右增大，y向下增大；\n"
    "每个框必须满足 x1 < x2 且 y1 < y2；confidence 为 0~1 的浮点数。\n"
    "若图片中没有检测到目标，boxes 输出空数组 []。"
)

SCHEMA_TEXT_POLY = (
    '输出 JSON 格式（必须严格符合，且不得输出 JSON 之外的任何内容）：\n'
    '{"description": "一句话图片描述", "objects": [{"name": "类别名", "points": [[x1, y1], [x2, y2], ...], "confidence": 0.9}]}\n'
    "坐标说明：均为像素坐标，原点(0,0)在图片左上角，x向右增大，y向下增大；\n"
    "points 为目标的轮廓多边形点序列（按轮廓顺序排列，至少 3 个点，8~40 个点为宜，尽量贴合目标边缘）；\n"
    "confidence 为 0~1 的浮点数。\n"
    "若图片中没有检测到目标，objects 输出空数组 []。"
)

DEFAULT_INSTRUCTION = (
    "请检测这张图片中的所有目标物体，逐个给出类别名称与边界框，"
    "并给出一句话整体图片描述。"
)


def default_api_config():
    return {"base_url": "http://127.0.0.1:1234/v1", "api_key": ""}


class ApiError(Exception):
    pass


class ApiCancel(Exception):
    """请求被用户主动取消（连接已断开）。"""


def _http_request(url, method, payload=None, timeout=300, abort_registry=None,
                  api_key=""):
    """发送 HTTP JSON 请求。abort_registry 为可选的列表：传入后本函数会把一个
    `abort()` 可调用对象追加进去，调用方可随时执行它以断开连接、取消本次请求
    （服务端检测到连接断开会立即终止推理）。"""
    parts = urllib.parse.urlsplit(url)
    host, port = parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)
    if parts.scheme == "https":
        conn = http.client.HTTPSConnection(host, port, timeout=timeout,
                                           context=ssl.create_default_context())
    else:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)

    def abort():
        try:
            conn.close()
        except Exception:
            pass

    if abort_registry is not None:
        abort_registry.append(abort)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    try:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        conn.request(method, parts.path + ("?" + parts.query if parts.query else ""),
                     body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
        if resp.status >= 400:
            raise ApiError("HTTP %s: %s" % (resp.status, data))
        return json.loads(data)
    except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError):
        raise ApiCancel("请求已取消（连接被断开）")
    except socket.timeout as e:
        raise ApiError("请求超时：服务 %d 秒未返回结果。大模型推理过慢时，"
                       "可降低并发或换用更快的模型。" % timeout)
    except (ConnectionError, TimeoutError, http.client.HTTPException) as e:
        raise ApiError("请求超时或连接失败: %s" % e)
    finally:
        if abort_registry is not None:
            try:
                abort_registry.remove(abort)
            except ValueError:
                pass
        conn.close()


# ---------------------------------------------------------------- 模型列表
def list_models(api_config=None):
    """通过 /v1/models 获取模型列表 [{name, size_gb, capabilities}]。"""
    api = api_config or default_api_config()
    base = api["base_url"].rstrip("/")
    data = _http_request(base + "/models", "GET", timeout=30,
                         api_key=api.get("api_key", ""))
    return [{
        "name": m.get("id"),
        "size_gb": 0.0,
        "capabilities": ["vision"],
    } for m in data.get("data", [])]


def list_vision_models(api_config=None):
    """OpenAI 兼容服务无法标记视觉能力，直接返回全部模型。"""
    return list_models(api_config)


# ---------------------------------------------------------------- 推理
def _image_mime(path):
    ext = os.path.splitext(path)[1].lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".webp": "image/webp", ".bmp": "image/bmp", ".gif": "image/gif",
        ".tif": "image/tiff", ".tiff": "image/tiff",
    }.get(ext, "image/jpeg")


def _b64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


SYSTEM_PROMPT = "你是计算机视觉数据标注助手。你必须只输出符合要求的 JSON，不要有任何多余内容。"


def _chat(model, prompt, image_path, temperature, max_tokens, use_format,
          abort_registry=None, api_config=None, thinking_off=True):
    # 每张图片 = 独立的全新对话：
    #  - messages 每次从头构造（系统提示 + 完整标注指令 + 本图），绝不携带上一张图片的任何
    #    history/context 字段，避免上下文累积串扰；无状态请求，服务端每次按新会话处理。
    api = api_config or default_api_config()
    mime = _image_mime(image_path)
    data_url = "data:%s;base64,%s" % (mime, _b64_image(image_path))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ],
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if use_format:
        payload["response_format"] = {"type": "json_object"}
    # qwen3 系列默认开启思考模式，会先输出大量思考内容、大幅拖慢响应；
    # 对批量检测标注任务无益，这里显式关闭（仅对 qwen3 非 VL 模型生效）。
    model_l = model.lower()
    if thinking_off and model_l.startswith("qwen3") and "-vl" not in model_l:
        payload["enable_thinking"] = False
    resp = _http_request(api["base_url"].rstrip("/") + "/chat/completions",
                         "POST", payload, abort_registry=abort_registry,
                         api_key=api.get("api_key", ""))
    return {"message": {"content": resp.get("choices", [{}])[0]
                        .get("message", {}).get("content", "")}}


def _parse_json(content):
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def _normalize_boxes(raw):
    """校验并规整模型返回的 boxes。"""
    boxes = []
    for b in raw.get("boxes", []) or []:
        if not isinstance(b, dict):
            continue
        try:
            x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
        except (KeyError, TypeError, ValueError):
            continue
        if x1 >= x2 or y1 >= y2:
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            if x1 == x2 or y1 == y2:
                continue
        conf = b.get("confidence", 1.0)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 1.0
        boxes.append({
            "name": str(b.get("name", "unknown")).strip() or "unknown",
            "x1": round(x1, 2), "y1": round(y1, 2),
            "x2": round(x2, 2), "y2": round(y2, 2),
            "confidence": round(min(max(conf, 0.0), 1.0), 4),
        })
    return boxes


def _normalize_polygons(data):
    """校验并规整模型返回的轮廓多边形。返回 [{name, points:[[x,y],...], confidence}]。"""
    objs = []
    for o in data.get("objects", []) or []:
        if not isinstance(o, dict):
            continue
        pts = []
        for p in o.get("points", []) or []:
            try:
                x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError, IndexError):
                continue
            pts.append([round(x, 1), round(y, 1)])
        if len(pts) < 3:
            continue
        conf = o.get("confidence", 1.0)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 1.0
        objs.append({
            "name": str(o.get("name", "unknown")).strip() or "unknown",
            "points": pts,
            "confidence": round(min(max(conf, 0.0), 1.0), 4),
        })
    return objs


def _chat_loop(model, prompt, image_path, temperature, max_tokens, retries,
               log, abort_registry, api_config, parse_fn):
    last_err = None
    thinking_off = True
    for attempt in range(retries):
        for use_format in (True, False) if attempt == 0 else (False,):
            try:
                resp = _chat(model, prompt, image_path, temperature, max_tokens,
                             use_format, abort_registry=abort_registry,
                             api_config=api_config, thinking_off=thinking_off)
                content = resp.get("message", {}).get("content", "")
                return parse_fn(_parse_json(content))
            except (json.JSONDecodeError, ValueError) as e:
                last_err = e
                if log:
                    log("JSON 解析失败(第 %d 次尝试)，将重新请求…" % (attempt + 1))
                continue
            except ApiError as e:
                msg = str(e)
                if thinking_off and "enable_thinking" in msg:
                    thinking_off = False
                    if log:
                        log("服务不支持关闭思考模式参数，已自动移除后重试…")
                    continue
                m = re.search(r"max_tokens\D*\[1,\s*(\d+)\]", msg)
                if m:
                    limit = int(m.group(1))
                    if max_tokens > limit:
                        max_tokens = limit
                        if log:
                            log("服务端 max_tokens 上限为 %d，已自动调低后重试…" % limit)
                        continue
                if "400" in str(e) and "format" in str(e).lower():
                    last_err = e
                    if log:
                        log("模型不支持 response_format=json_object，切换宽松模式重试…")
                    continue
                raise
        if attempt < retries - 1:
            prompt = prompt + "\n\n注意：你上次的输出不是合法 JSON，请只输出严格合法的 JSON。"
    raise ValueError("模型连续 %d 次未返回合法 JSON：%s" % (retries, last_err))


def chat_json(model, prompt, image_path, temperature=0.1, max_tokens=4096,
              retries=3, log=None, abort_registry=None, api_config=None):
    """发送图片+指令，返回解析后的 {description, boxes}。失败抛 ApiError/ValueError；
    被取消时抛 ApiCancel。abort_registry 传入可中止本次请求的注册表（见 _http_request）。"""
    def parse(data):
        return {
            "description": str(data.get("description", "")).strip(),
            "boxes": _normalize_boxes(data),
        }
    return _chat_loop(model, prompt, image_path, temperature, max_tokens,
                      retries, log, abort_registry, api_config, parse)


def chat_polygons(model, prompt, image_path, temperature=0.1, max_tokens=16384,
                  retries=3, log=None, abort_registry=None, api_config=None):
    """实例分割模式：模型直接输出轮廓多边形。返回
    {description, objects: [{name, points, confidence}]}。"""
    def parse(data):
        return {
            "description": str(data.get("description", "")).strip(),
            "objects": _normalize_polygons(data),
        }
    return _chat_loop(model, prompt, image_path, temperature, max_tokens,
                      retries, log, abort_registry, api_config, parse)