# 图像自动标注工具

基于 OpenAI 兼容视觉 API 的批量图像自动标注桌面工具。拖入图片 → 视觉模型按自定义指令推理（目标检测 / 实例分割）→ 一键导出 JSON / YOLO TXT / COCO / Markdown 四种格式到独立目录。

## 功能特性

- **批量拖拽**：图片或整个文件夹直接拖入，自动递归收集
- **自定义指令**：任意标注指令（如"检测所有葡萄，标注品种与成熟度"），默认内置示例
- **双后端 API**：支持任意 OpenAI 兼容服务（LM Studio、vLLM、Ollama `/v1`、阿里云百炼 DashScope、OpenAI 等），Base URL + API Key 配置即用，模型列表自动拉取或手动填写
- **两种标注模式**：
  - 目标检测（边界框）：VL 模型直接输出 `{description, boxes}`
  - 实例分割（轮廓）：二选一引擎
    - **API 多边形**：模型直接输出轮廓点序列，零依赖
    - **本地 SAM2**：VL 出检测框 → 框中心提示 SAM2 → 像素级精确掩码（质量最佳）
- **四格式独立导出**：
  | 格式 | 内容 |
  |---|---|
  | JSON | 每图一个 `.json`（描述 + 框/多边形 + 置信度） |
  | YOLO TXT | 每图 `.txt` + `classes.txt`；实例分割时自动切换 YOLO-seg 格式（class cx cy w h + 归一化多边形点） |
  | COCO | 单文件 `annotations.json`，含 `segmentation` 多边形 |
  | Markdown | 图文报告 `.md`（图片自动复制），含轮廓点数列 |
- **工程化细节**：并发批处理（可调）、一键停止（中断在途请求）、大图自动缩放（坐标换算回原图）、max_tokens 超限自动降级、JSON 解析容错重试、失败图片空标注兜底、设置持久化
- **智能诊断**：空检测结果时输出模型描述，快速分辨"图中无目标"与"模型无视觉能力"

## 环境要求

| 组件 | 要求 |
|---|---|
| Python | 3.10+（主环境） |
| PySide6 | 6.8.x（6.11 在部分 Win 环境有 DLL 加载问题） |
| 视觉模型 API | 任一 OpenAI 兼容服务（可联网，或本地 LM Studio / Ollama） |
| SAM2（可选） | 独立 conda 环境，`torch`（CUDA 版）+ `sam2` + `opencv`；NVIDIA GPU 推荐 |

## 安装与启动

```bash
# 1. 安装依赖（主环境）
pip install PySide6==6.8.3

# 2. 启动
python main.py
# 或 Windows 下双击 启动.bat
```

配置连接：填入 Base URL（如阿里云百炼 `https://dashscope.aliyuncs.com/compatible-mode/v1`）与 API Key，点"刷新"拉取模型列表（也可手动输入模型名）。选择支持**图片输入**的视觉模型（如 `qwen-vl-max`、`qwen3.7-plus`、`gpt-4o` 等）。

## 实例分割：SAM2 本地引擎

### 环境搭建（一次性）

```bash
# 创建 conda 环境（示例名 grape_seg）
conda create -n grape_seg python=3.10 -y
conda activate grape_seg

# 安装 CUDA 版 PyTorch（NVIDIA 显卡；CPU 版可去掉 --index-url 并换 cpu 包）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install sam2 opencv-python pillow

# 下载 SAM2.1 tiny 权重到 checkpoints/ 目录（约 148MB；国内可用 hf-mirror 镜像）
python -c "import urllib.request; urllib.request.urlretrieve('https://hf-mirror.com/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_tiny.pt', 'checkpoints/sam2.1_hiera_tiny.pt')"
```

### 使用

界面"标注类型"选"实例分割" → 分割引擎选"本地 SAM2" → 填写 SAM2 环境 Python 路径（如 `D:\Anaconda\envs\grape_seg\python.exe`）。

流程：VL 模型检测框（并发）→ SAM2 对每框中心点做像素分割（原图精度，串行 GPU 推理）→ 多边形简化后导出。SAM2 进程为常驻服务，模型仅加载一次（约 3~4s），崩溃自动重启。

## 导出格式示例

**YOLO-seg**（`类id cx cy w h x1 y1 x2 y2 ...`，均归一化）：

```
0 0.500000 0.500000 0.562500 0.625000 0.219 0.188 0.281 0.156 ...
```

**COCO**：`segmentation` 为多边形扁平数组 `[x1, y1, x2, y2, ...]`；无分割数据时退回 bbox 四点多边形。

## 项目结构

```
├── main.py          # GUI 入口（PySide6）+ 批处理调度
├── api_client.py    # OpenAI 兼容 API 客户端（检测/多边形两种请求，容错重试）
├── exporters.py     # JSON / YOLO / COCO / Markdown 导出器
├── sam_cli.py       # SAM2 常驻子进程服务（stdin/stdout JSON 行协议）
├── checkpoints/     # SAM2 权重（不入库，见上）
└── test_*.py        # 回归测试脚本（offscreen 运行）
```

## 回归测试

```bash
python test_gui.py      # GUI 全流程 + 四格式导出
python test_stop.py     # 停止按钮中断
python test_seg.py      # 实例分割全链路（SAM2 真实分割）
python test_seg_api.py  # API 多边形模式
```

## 常见问题

- **模型返回 400 "messages.0.role"**：选到了文生图/纯文本模型（如 `qwen-image-*`、开源无视觉版），请换用视觉理解模型
- **"成功"但导出为空**：模型看不到图片（无视觉输入）或图中无目标；日志中的模型描述可帮助判断
- **max_tokens 超限**：程序自动钳制到服务端允许的上限并重试
