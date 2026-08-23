# SAM3easyuse

基于 [SAM3 (Segment Anything Model 3)](https://github.com/facebookresearch/sam3) 的开箱即用推理工具，提供 **Gradio Web UI** 和 **Python/CLI 接口**，适配 Windows + CUDA 环境。

## 功能

### 图片处理

| 模式        | 说明                              |
| ----------- | --------------------------------- |
| 📝 文本分割 | 输入文字描述，自动检测并分割目标  |
| 🔲 框选分割 | 画框标记区域，可结合文本提示      |
| 👆 点击分割 | 点击标记前景/背景点，交互式分割   |
| 📦 批量分割 | 文件夹批量处理 / 视频拆帧批量处理 |

### 视频跟踪

| 模式        | 说明                                 |
| ----------- | ------------------------------------ |
| 📝 文本跟踪 | 文本描述目标，全视频自动跟踪         |
| 👆 点击跟踪 | 在任意帧点击标记，向前后传播跟踪     |
| 🔲 框选跟踪 | 框选目标区域，可结合文本，全视频跟踪 |

### 通用特性

- 支持 **SAM3 / SAM3.1** 模型切换
- 支持 **Flash Attention 2** 开关（加速推理）
- 视频跟踪支持 **自选中间帧** 标注（非仅首帧）
- 输出支持 **叠加可视化** 和 **二值 Mask** 两种模式
- 附带 [mask 二次加工 + HTTP API 范例](#范例活用-mask-做二次加工并封装为-api)（`face_grid_api.py`）

## 环境要求

- Windows x64
- Python 3.12
- CUDA 12.8（需要 NVIDIA GPU）
- FFmpeg（用于视频编码）

## 安装

### 1. 克隆项目

```bash
git clone --recursive https://github.com/RakuhaSociety/SAM3easyuse.git
cd SAM3easyuse
```

> `--recursive` 会自动拉取 sam3 子模块。如果忘了加，后续执行：
>
> ```bash
> git submodule update --init --recursive
> ```

### 2. 创建 Python 环境

```bash
conda create -p SAM3easyuse_env python=3.12 -y
```

### 3. 安装 PyTorch（CUDA 12.8）

```bash
SAM3easyuse_env\python.exe -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
```

### 4. 安装 Triton

```bash
# Windows
SAM3easyuse_env\python.exe -m pip install triton-windows

# Linux
SAM3easyuse_env/bin/python -m pip install triton
```

### 5. 安装 Flash Attention 2

从 [Release](https://github.com/RakuhaSociety/SAM3easyuse/releases) 下载预编译 whl：

```bash
SAM3easyuse_env\python.exe -m pip install flash_attn-2.8.3+cu128torch2.9.0cxx11abiTRUE-cp312-cp312-win_amd64.whl
```

> 如果你的环境不匹配此 whl，可以在 Gradio UI / CLI 中关闭 Flash Attention（`--no-fa`），程序会回退到 SDPA。

### 6. 安装其余依赖

```bash
SAM3easyuse_env\python.exe -m pip install -r requirements.txt
```

### 7. 安装 SAM3

```bash
cd sam3
..\SAM3easyuse_env\python.exe -m pip install -e .
cd ..
```

### 8. 下载模型权重

将以下文件放入 `sam3/checkpoints/` 目录：

- `sam3.pt` — SAM3 模型
- `sam3.1_multiplex.pt` — SAM3.1 模型

### 9. FFmpeg

将 FFmpeg 解压到项目根目录（或确保 `ffmpeg` 在系统 PATH 中）。

## 使用

### Gradio Web UI

```bash
SAM3easyuse_env\python.exe gradio_app.py
```

浏览器打开 `http://localhost:7860`。

### CLI 命令行

```bash
# 图像文本分割
python inference.py image-text -i photo.jpg -t "person, car" -o result.png

# 图像框选分割
python inference.py image-box -i photo.jpg --box 100,50,400,300 -o result.png

# 图像点击分割
python inference.py image-points -i photo.jpg --points 200,150,1 350,200,0 -o result.png

# 批量分割（文件夹）
python inference.py batch -d ./images -t "person" -o ./results

# 批量分割（视频拆帧）
python inference.py batch -v input.mp4 -t "car" --interval 5 -o ./results

# 视频文本跟踪
python inference.py video-text -v input.mp4 -t "person" -o tracked.mp4

# 视频点击跟踪
python inference.py video-points -v input.mp4 --points 200,150,1 --frame 30 -o tracked.mp4

# 视频框选跟踪（支持多框 + 正/负向框，可选加文本）
python inference.py video-box -v input.mp4 --box 100,50,400,300 -t "person" -o tracked.mp4

# 多个正向框 + 负向框排除（例：跟踪人脸但排除画面左上角的脸）
python inference.py video-box -v input.mp4 --box 100,50,400,300 --neg-box 10,10,80,80 -t "face" -o tracked.mp4
```

通用选项：

- `--model sam3.1` — 使用 SAM3.1 模型
- `--mask` — 输出二值 Mask
- `--no-fa` — 禁用 Flash Attention
- `--mmgp` — 启用 mmgp 显存优化
- `--mmgp-profile N` — mmgp profile，1–5，默认 4
- `--sam31-batch-size N` — SAM3.1 视频 backbone 批大小（CLI 默认 4；官方默认 16，启用 mmgp 建议 1）

### 作为 Python 库

```python
from inference import SAM3Inference

sam = SAM3Inference(version="sam3.1", use_fa=True)

# 图片分割
result, info = sam.segment_image_text("photo.jpg", "person, car")
result, info = sam.segment_image_box("photo.jpg", boxes=[(100, 50, 400, 300)])
result, info = sam.segment_image_points("photo.jpg", points=[(200, 150, 1)])

# 视频跟踪
path, info = sam.track_video_text("input.mp4", "person")
path, info = sam.track_video_points("input.mp4", [(200, 150, 1)], frame_idx=30)

# 视频框选跟踪：单框（旧 API 兼容）
path, info = sam.track_video_box("input.mp4", (100, 50, 400, 300), text="person")
# 视频框选跟踪：多框 + 负向框（每项 (x1,y1,x2,y2[,is_pos])）
path, info = sam.track_video_box(
    "input.mp4",
    boxes=[(100, 50, 400, 300, True), (10, 10, 80, 80, False)],
    text="face",
)

sam.unload_all()  # 释放显存
```

启用 mmgp 时（SAM3.1 视频推荐设 `sam31_batch_size=1` 以最大化节省显存）：

```python
sam = SAM3Inference(
    version="sam3.1",
    use_fa=True,
    use_mmgp=True,
    mmgp_profile=4,
    sam31_batch_size=1,   # mmgp 下逐帧运行，显存需求最低
)
```

## 范例：活用 mask 做二次加工并封装为 API

`face_grid_api.py` 是一个完整的实战范例，演示两件在文档里不太好讲清楚的事：**怎么把 SAM 输出的 mask 真正用起来**，以及**怎么把 SAM 包成一个常驻 HTTP 服务**。它的业务功能是给检测到的目标叠加网格线（常用于人脸打码），但拆开看，里面的模式可以直接套用到任何"分割 + 后处理"的需求上。

### 一、mask 不只是拿来抠图

大多数示例到 `masks` 就结束了。这个范例展示 mask 作为**布尔蒙版参与像素运算**的用法，核心只有三步（见 [face_grid_api.py:410-424](face_grid_api.py#L410-L424)）：

```python
# 1. 多目标 mask 合并成一张：masks 形状 (N, 1, H, W) → (H, W) 的 bool
combined_mask = torch.any(masks.squeeze(1), dim=0).cpu().numpy()

# 2. 生成一个与图同尺寸的程序化图案（这里是任意角度的交叉网格）
grid_pattern = create_grid_pattern(h, w, adj_lw, adj_ls, angle)

# 3. 布尔与运算求交集 —— 图案只在目标区域内生效，背景不受影响
apply_region = combined_mask & grid_pattern
frame_out[apply_region] = 0
```

关键在第 3 步。`mask & pattern` 把"哪里是目标"和"画什么效果"彻底解耦：换掉 `create_grid_pattern`，就能变成马赛克、高斯模糊、纯色填充、贴图，而定位逻辑完全不用动。想反向操作（只处理背景、保留主体）也只需改成 `~combined_mask & pattern`。

另外两个容易踩的点范例里也处理了：

- **尺寸对不齐**：mask 分辨率不一定等于原图，合并后要用 `INTER_NEAREST` 缩放回原尺寸（[:414-418](face_grid_api.py#L414-L418)）。用默认的双线性插值会让 bool 边缘产生中间值，蒙版就糊了。
- **根据目标大小自适应参数**：`inference_state["boxes"]` 里的检测框可以拿来反推目标尺度，进而动态调整效果强度。范例中 [compute_face_scale()](face_grid_api.py#L189) 让小脸用密网格、大脸用疏网格，避免固定参数在不同构图下失效。

### 二、封装成 API 的几个要点

直接跑起来：

```bash
python face_grid_api.py --port 8000 --version 3.0
```

启动后 `http://127.0.0.1:8000/docs` 有自动生成的交互式文档。

> **注意监听地址。** `--host` 默认 `0.0.0.0`，即绑定所有网卡，局域网内任何人都能访问该端口上传文件并占用你的 GPU。服务本身没有认证机制，仅供内部或可信网络使用。只在本机调用时请显式指定 `--host 127.0.0.1`；需要对外提供服务时，在前面挂一层带鉴权的反向代理，不要直接暴露。

Windows 下每次手动配环境变量比较烦，可以按需自建一个启动脚本（本地脚本已被 `.gitignore` 排除，不随仓库分发）。在项目根目录新建 `启动API服务.bat`：

```bat
@echo off
chcp 65001 >nul

set PYTHON=%CD%\SAM3mmgp_env\python.exe
set FF_PATH=%CD%\ffmpeg-8.1-full_build-shared\bin
set CONDA_LIB=%CD%\SAM3mmgp_env\Library\bin
set CU_PATH=%CD%\SAM3mmgp_env\Lib\site-packages\torch\lib
set SC_PATH=%CD%\SAM3mmgp_env\Scripts
set PATH=%FF_PATH%;%CONDA_LIB%;%CU_PATH%;%SC_PATH%;%PATH%
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=%CD%\.huggingface
set TORCH_HOME=%CD%\.huggingface
set XFORMERS_FORCE_DISABLE_TRITON=1
set FFMPEG_PATH=%CD%\ffmpeg-8.1-full_build-shared\bin

%PYTHON% face_grid_api.py ^
--host 127.0.0.1 ^
--port 8000 ^
--version 3.0

pause
```

其中 `PATH` 那几行是关键：ffmpeg 和 torch 的 DLL 目录必须在 Python 启动前就进 `PATH`，否则视频编码或 CUDA 初始化会失败。目录名按你自己的环境和 ffmpeg 版本调整。

| 端点             | 说明                                |
| ---------------- | ----------------------------------- |
| `POST /process`  | 按文件扩展名自动分发到图片或视频    |
| `POST /process/image` | 图片处理                       |
| `POST /process/video` | 视频处理（逐帧跟踪）           |
| `GET /health`    | 健康检查，返回模型加载状态          |

表单参数：`prompt`（分割目标，默认 `face`）、`line_width`、`line_spacing`、`angle`。

```bash
# 默认分割人脸
curl -X POST http://127.0.0.1:8000/process/image   -F "file=@photo.jpg" -o result.jpg

# 换任意目标：prompt 接受任意名词短语
curl -X POST http://127.0.0.1:8000/process/image   -F "file=@street.jpg" -F "prompt=truck" -o result.jpg

# 调整网格外观
curl -X POST http://127.0.0.1:8000/process/video   -F "file=@input.mp4" -F "prompt=person"   -F "line_width=3" -F "line_spacing=12" -F "angle=30" -o result.mp4
```

范例中值得照搬的几个做法：

**模型常驻，不要每个请求重新加载。** 权重加载是秒级甚至十秒级开销，服务端必须复用。范例在启动时把图片模型和视频模型各加载成一个全局单例（`_image_processor` / `_predictor`，[:54-58](face_grid_api.py#L54-L58)），请求只做推理。

**并发要串行化，但别阻塞事件循环。** SAM 的 `inference_state` 是有状态的，多请求并发写同一个 processor 会互相污染，所以推理段用 `threading.Lock` 保护（[:243](face_grid_api.py#L243)、[:373](face_grid_api.py#L373)），同一个模型同时只跑一个任务 —— 这不是性能妥协，是正确性要求。同时端点是 `async def`，推理必须用 `asyncio.to_thread` 丢到线程池（[:480](face_grid_api.py#L480)、[:540](face_grid_api.py#L540)），否则几十秒的同步推理会卡死整个服务，连 `/health` 都不响应。

**每次推理后重置提示词。** `reset_all_prompts(inference_state)` 必须调用（[:411](face_grid_api.py#L411)），否则上一次请求的文本提示会残留到下一次，表现为"传了新 prompt 但结果还是旧目标"。

**提示词失败要有回退。** 文本分割对措辞敏感，`face` 有时无结果而 `human face` 有。范例用 [build_prompt_candidates()](face_grid_api.py#L180) 按优先级依次尝试，命中即止；且只对人体类提示词追加 `human` 前缀 —— 给 `car` 加前缀变成 `human car` 毫无意义。

**没检测到目标时的行为要明确区分。** 图片端点返回原图（打码场景下"没脸可打"是正常结果），视频端点返回 `422` 并说明未命中的提示词（整段视频都没有目标，更可能是提示词写错了）。

**临时文件用 `BackgroundTask` 清理。** 响应体还没发完就删文件会截断传输，挂在后台任务里等发送完成后再删。

## 项目结构

```
SAM3easyuse/
├── gradio_app.py        # Gradio Web UI
├── inference.py         # SAM3Inference 类 + CLI
├── face_grid_api.py     # 范例：mask 二次加工 + FastAPI 服务
├── requirements.txt     # Python 依赖
├── sam3/                # SAM3 源码 (git submodule)
│   ├── checkpoints/     # 模型权重 (需自行下载)
│   └── ...
└── outputs/             # 推理结果输出目录
```

## mmgp 显存优化

[mmgp (Memory Management for the GPU Poor)](https://github.com/deepbeepmeep/mmgp) 可将模型权重分片管理在 RAM 与 VRAM 之间，显著降低峰值显存占用。

### 安装

```bash
pip install mmgp
```

### 在 Gradio UI 中启用

启动 Web UI 后，顶部设置栏勾选 **💾 mmgp 显存优化**，并选择合适的 Profile：

| Profile                      | 适合场景                  | RAM 要求 | VRAM 要求 |
| ---------------------------- | ------------------------- | -------- | --------- |
| 1 - HighRAM_HighVRAM         | 最快，批量短视频          | ≥ 48 GB | ≥ 24 GB  |
| 2 - HighRAM_LowVRAM          | RTX 3080/4080 推荐        | ≥ 48 GB | ≥ 12 GB  |
| 3 - LowRAM_HighVRAM          | RAM 有限但 VRAM 充足      | ≥ 32 GB | ≥ 24 GB  |
| **4 - LowRAM_LowVRAM** | **默认，12GB 显卡** | ≥ 32 GB | ≥ 12 GB  |
| 5 - VeryLowRAM_LowVRAM       | 最省显存                  | ≥ 24 GB | ≥ 10 GB  |

> 启用后下次调用加载模型时自动生效；切换 Profile 后无须重启，下次推理生效。

### 在 CLI 中启用

```bash
# 图像分割 + mmgp profile 4（默认）
python inference.py image-text -i photo.jpg -t "person" --mmgp

# 指定 profile 2（HighRAM_LowVRAM，速度更快）
python inference.py image-text -i photo.jpg -t "person" --mmgp --mmgp-profile 2

# 视频跟踪（视频模型 mmgp 为 best-effort 模式，效果取决于模型结构）
python inference.py video-text -v input.mp4 -t "person" --mmgp
```

### 作为 Python 库使用

```python
from inference import SAM3Inference

# 图像推理 + mmgp
sam = SAM3Inference(
    version="sam3.1",
    use_fa=True,
    use_mmgp=True,
    mmgp_profile=4,
    sam31_batch_size=1,  # mmgp 下逐帧运行节省显存
)
result, info = sam.segment_image_text("photo.jpg", "person")

# 视频推理 + mmgp（SAM3.1）
sam = SAM3Inference(
    version="sam3.1",
    use_fa=True,
    use_mmgp=True,
    mmgp_profile=4,
    sam31_batch_size=1,
)
path, info = sam.track_video_text("input.mp4", "person")
```

### 注意事项

- **图像模型**（文本分割、框选分割、点击分割）完整支持 mmgp
- **视频模型**支持 mmgp，SAM3 和 SAM3.1 均已验证可用
- Windows 系统比 Linux 额外需要约 16 GB RAM
- mmgp 会对 `transformer` 组件进行 8-bit 量化，可能导致精度略有下降

## 致谢

- [SAM3 — Meta AI Research](https://github.com/facebookresearch/sam3)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [Flash Attention Windows Builds](https://github.com/bdashore3/flash-attention)

## License

本项目工具代码采用 MIT License。SAM3 模型代码遵循其[原始许可](sam3/LICENSE)。
