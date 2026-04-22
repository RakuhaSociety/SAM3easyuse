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
SAM3easyuse_env\python.exe -m pip install torch==2.9.0+cu128 torchvision==0.24.0+cu128 torchaudio==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

### 4. 安装 Flash Attention 2

从 [Release](https://github.com/RakuhaSociety/SAM3easyuse/releases) 下载预编译 whl：

```bash
SAM3easyuse_env\python.exe -m pip install flash_attn-2.8.3+cu128torch2.9.0cxx11abiTRUE-cp312-cp312-win_amd64.whl
```

> 如果你的环境不匹配此 whl，可以在 Gradio UI / CLI 中关闭 Flash Attention（`--no-fa`），程序会回退到 SDPA。

### 5. 安装其余依赖

```bash
SAM3easyuse_env\python.exe -m pip install -r requirements.txt
```

### 6. 安装 SAM3

```bash
cd sam3
..\SAM3easyuse_env\python.exe -m pip install -e .
cd ..
```

### 7. 下载模型权重

将以下文件放入 `sam3/checkpoints/` 目录：

- `sam3.pt` — SAM3 模型
- `sam3.1_multiplex.pt` — SAM3.1 模型

### 8. FFmpeg

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

# 视频框选跟踪（可选加文本）
python inference.py video-box -v input.mp4 --box 100,50,400,300 -t "person" -o tracked.mp4
```

通用选项：

- `--model sam3.1` — 使用 SAM3.1 模型
- `--mask` — 输出二值 Mask
- `--no-fa` — 禁用 Flash Attention
- `--mmgp` — 启用 mmgp 显存优化
- `--mmgp-profile N` — mmgp profile，1–5，默认 4
- `--sam31-batch-size N` — SAM3.1 视频 backbone 批大小（默认 1，mmgp 模式建议 4）

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
path, info = sam.track_video_box("input.mp4", (100, 50, 400, 300), text="person")

sam.unload_all()  # 释放显存
```

启用 mmgp 时（SAM3.1 视频推荐设 `sam31_batch_size=4`）：

```python
sam = SAM3Inference(
    version="sam3.1",
    use_fa=True,
    use_mmgp=True,
    mmgp_profile=4,
    sam31_batch_size=4,   # 批量 grounding，提升视频推理速度
)
```

## 项目结构

```
SAM3easyuse/
├── gradio_app.py        # Gradio Web UI
├── inference.py         # SAM3Inference 类 + CLI
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
    use_mmgp=True,       # 启用 mmgp
    mmgp_profile=4,      # 1-5，默认 4
)
result, info = sam.segment_image_text("photo.jpg", "person")

# 视频推理 + mmgp + 批处理加速（SAM3.1）
sam = SAM3Inference(
    version="sam3.1",
    use_fa=True,
    use_mmgp=True,
    mmgp_profile=4,
    sam31_batch_size=4,  # 批量 backbone 推理，提升吞吐
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
