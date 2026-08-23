"""
FastAPI 服务：对视频/图片中的人头或人体提取蒙版并叠加交叉黑色网格线。

启动方式:
    python face_grid_api.py [--host 0.0.0.0] [--port 8000] [--version 3.0]

API:
    POST /process/video  — 上传视频，返回处理后的视频
    POST /process/image  — 上传图片，返回处理后的图片
    GET  /health         — 健康检查

    表单参数 (两个端点通用):
        file         - 视频/图片文件 (必需)
        prompt       - 分割目标提示词，默认 "face"
                       (任意名词短语，如 person / glasses / truck / dog；
                        人体类提示词检测失败时会自动重试 "human <prompt>")
        line_width   - 网格线宽度，默认 2
        line_spacing - 网格线间距，默认 9
        angle        - 网格线角度 0-90，默认 45
"""

import argparse
import os
import sys
import tempfile
import threading
import uuid

import cv2
import numpy as np
import torch

try:
    from mmgp import offload as _mmgp_offload
    from mmgp import profile_type as _mmgp_profile_type
    _MMGP_AVAILABLE = True
except Exception:
    _mmgp_offload = None
    _mmgp_profile_type = None
    _MMGP_AVAILABLE = False

# 将 sam3 仓库加入 path
SAM3_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3")
sys.path.insert(0, SAM3_ROOT)

CHECKPOINTS = {
    "3.0": os.path.join(SAM3_ROOT, "checkpoints", "sam3.pt"),
    "3.1": os.path.join(SAM3_ROOT, "checkpoints", "sam3.1_multiplex.pt"),
}

BPE_PATH = os.path.join(SAM3_ROOT, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")

# 全局预测器（模型只加载一次）
_predictor = None       # 视频预测器
_predictor_lock = threading.Lock()

_image_processor = None  # 图片处理器
_image_lock = threading.Lock()


def get_predictor():
    return _predictor


def build_predictor(checkpoint_path, version="3.0"):
    """构建 SAM3 视频预测器"""
    print(f"正在加载模型 (SAM {version}): {checkpoint_path}")
    if version == "3.1":
        from sam3.model_builder import build_sam3_multiplex_video_predictor
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=checkpoint_path,
            async_loading_frames=False,
        )
    else:
        from sam3.model_builder import build_sam3_video_predictor
        predictor = build_sam3_video_predictor(
            checkpoint_path=checkpoint_path,
            gpus_to_use=[0],
            async_loading_frames=False,
        )
    print("模型加载完成")
    return predictor


def _apply_mmgp(module, target_name: str, module_key: str = "model", **override_kwargs):
    """对模块应用 mmgp VRAM 卸载."""
    if not _MMGP_AVAILABLE or module is None:
        return
    profile_name = "LowRAM_LowVRAM"
    profile = getattr(_mmgp_profile_type, profile_name, None)
    if profile is None:
        for name in dir(_mmgp_profile_type):
            if not name.startswith("_"):
                profile = getattr(_mmgp_profile_type, name)
                break
    if profile is None:
        print("[MMGP] 未找到可用 profile，跳过")
        return
    try:
        _mmgp_offload.profile(module, profile_no=profile, **override_kwargs)
    except Exception:
        try:
            _mmgp_offload.profile({module_key: module}, profile_no=profile, **override_kwargs)
        except Exception as e:
            print(f"[MMGP] 应用失败 ({target_name}): {e}")
            return
    print(f"[MMGP] 已应用到 {target_name}")
    try:
        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")
    except Exception:
        pass


def _cleanup_gpu():
    """释放 GPU 缓存 + pinned host memory（对齐 inference.py）.
    mmgp 卸载后 pinned host memory 不会随 torch.cuda.empty_cache() 释放，
    需通过 _mmgp_offload.flush_torch_caches() 一站式处理。
    """
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if _MMGP_AVAILABLE:
        try:
            _mmgp_offload.flush_torch_caches()
            return
        except Exception:
            pass
    try:
        torch._C._host_emptyCache()
    except AttributeError:
        pass
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            PROCESS_SET_QUOTA = 0x0100
            PROCESS_QUERY_INFORMATION = 0x0400
            handle = kernel32.OpenProcess(
                PROCESS_SET_QUOTA | PROCESS_QUERY_INFORMATION, False, os.getpid()
            )
            if handle:
                psapi.EmptyWorkingSet(handle)
                kernel32.CloseHandle(handle)
        except Exception:
            pass


def create_grid_pattern(h, w, line_width, line_spacing, angle_deg):
    """生成交叉网格图案，返回 bool 数组"""
    angle_rad = np.deg2rad(angle_deg)
    nx1 = np.cos(angle_rad)
    ny1 = np.sin(angle_rad)
    nx2 = np.cos(-angle_rad)
    ny2 = np.sin(-angle_rad)

    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    period = line_width + line_spacing

    proj1 = x_coords * nx1 + y_coords * ny1
    on_line1 = (proj1 % period) < line_width

    proj2 = x_coords * nx2 + y_coords * ny2
    on_line2 = (proj2 % period) < line_width

    return on_line1 | on_line2


# 与人体相关的提示词，检测失败时可退化为 "human <prompt>" 再试一次；
# 其他目标（如 car、dog）加 human 前缀没有意义，不做该回退。
HUMAN_PROMPTS = {
    "face", "head", "hair", "eye", "eyes", "nose", "mouth", "ear", "ears",
    "hand", "hands", "arm", "leg", "body", "person", "people",
    "glasses", "hat", "cap", "mask", "helmet",
}


def build_prompt_candidates(prompt):
    """返回按优先级排列的提示词候选列表。"""
    prompt = (prompt or "").strip() or "face"
    candidates = [prompt]
    if prompt.lower() in HUMAN_PROMPTS:
        candidates.append(f"human {prompt}")
    return candidates


def compute_face_scale(face_sizes_px, image_size, reference_fraction=0.030):
    """
    Compute scale factor for line_width and line_spacing based on face size.

    At reference_fraction (face_max_dim / image_max_dim = 0.15), scale = 1.0.
    Smaller faces → scale < 1 (denser grid), larger faces → scale > 1 (sparser grid).

    Args:
        face_sizes_px: iterable of face sizes (max(width, height)) in pixels
        image_size: max(image_width, image_height)
        reference_fraction: face-to-image ratio that maps to scale = 1.0

    Returns:
        scale factor, clamped to [0.3, 3.0]
    """
    if not face_sizes_px:
        return 1.0
    avg_face_size = sum(float(s) for s in face_sizes_px) / len(face_sizes_px)
    scale = avg_face_size / (image_size * reference_fraction)
    return max(0.3, min(3.0, scale))


def process_video(input_path, output_path, line_width, line_spacing, angle, prompt="face"):
    """
    对视频进行目标蒙版提取 + 网格线叠加，写入 output_path。
    使用全局 predictor，加锁保证同时只处理一个视频。

    Args:
        prompt: 分割目标提示词（如 face, person, truck 等），默认 "face"
    """
    predictor = get_predictor()

    # 读取视频信息和帧
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_bgr = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
    cap.release()

    if not frames_bgr:
        raise ValueError("视频没有可读取的帧")

    print(f"  分辨率: {width}x{height}, FPS: {fps:.1f}, 总帧数: {len(frames_bgr)}")

    with _predictor_lock, torch.autocast("cuda", dtype=torch.bfloat16):
        # 启动会话
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=input_path,
                offload_video_to_cpu=True,
            )
        )
        session_id = response["session_id"]

        # 在多个帧上尝试检测目标
        total_frames = len(frames_bgr)
        # 采样帧索引：第 0 帧 + 均匀分布的若干帧
        sample_count = min(10, total_frames)
        sample_indices = sorted(set(
            [0] + [int(i * total_frames / sample_count) for i in range(1, sample_count)]
        ))

        n_objects = 0
        prompt_frame_index = 0
        object_scale = 1.0
        for text_prompt in build_prompt_candidates(prompt):
            for fidx in sample_indices:
                predictor.handle_request(
                    request=dict(type="reset_session", session_id=session_id)
                )
                response = predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=fidx,
                        text=text_prompt,
                    )
                )
                out = response["outputs"]
                n_objects = len(out.get("out_obj_ids", []))
                if n_objects > 0:
                    prompt_frame_index = fidx
                    boxes_norm = out.get("out_boxes_xywh", [])
                    if len(boxes_norm) > 0:
                        box_sizes = [max(b[2] * width, b[3] * height) for b in boxes_norm]
                        object_scale = compute_face_scale(box_sizes, max(width, height))
                    print(f"  在第 {fidx} 帧 (提示词: '{text_prompt}') 检测到 {n_objects} 个 {prompt}, scale={object_scale:.2f}")
                    break
            if n_objects > 0:
                break

        if n_objects == 0:
            predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )
            _cleanup_gpu()
            raise ValueError(f"未检测到目标: {prompt}")

        # 传播蒙版
        masks_per_frame = {}
        for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction="both",
            )
        ):
            frame_idx = response["frame_index"]
            outputs = response["outputs"]
            if "out_binary_masks" in outputs and len(outputs["out_binary_masks"]) > 0:
                combined_mask = np.any(outputs["out_binary_masks"], axis=0)
                masks_per_frame[frame_idx] = combined_mask
            else:
                masks_per_frame[frame_idx] = np.zeros((height, width), dtype=bool)

        # 关闭会话
        predictor.handle_request(
            request=dict(type="close_session", session_id=session_id)
        )

        # 释放 GPU 显存缓存 + pinned host memory
        _cleanup_gpu()

    # 生成网格图案
    adj_lw = max(1, int(line_width * object_scale))
    adj_ls = max(1, int(line_spacing * object_scale))
    grid_pattern = create_grid_pattern(height, width, adj_lw, adj_ls, angle)
    print(f"  网格参数: line_width={adj_lw}, line_spacing={adj_ls} (scale={object_scale:.2f})")

    # 写入输出视频
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(len(frames_bgr)):
        frame = frames_bgr[i]
        mask = masks_per_frame.get(i, np.zeros((height, width), dtype=bool))

        if mask.shape[:2] != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8), (width, height),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        apply_region = mask & grid_pattern
        frame_out = frame.copy()
        frame_out[apply_region] = 0
        writer.write(frame_out)

    writer.release()
    print(f"  输出完成: {output_path}")


def process_image(input_path, output_path, line_width, line_spacing, angle, prompt="face"):
    """
    对图片进行目标蒙版提取 + 网格线叠加，写入 output_path。

    Args:
        prompt: 分割目标提示词（如 face, person, truck 等），默认 "face"
    """
    from PIL import Image as PILImage

    processor = _image_processor
    if processor is None:
        raise ValueError("图片模型未加载")

    image_cv = cv2.imread(input_path)
    if image_cv is None:
        raise ValueError(f"无法读取图片: {input_path}")

    image_pil = PILImage.open(input_path).convert("RGB")
    h, w = image_cv.shape[:2]
    object_scale = 1.0

    with _image_lock, torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = processor.set_image(image_pil)

        # 依次尝试候选提示词，命中即止（set_text_prompt 会覆盖上一次的文本提示）
        masks = None
        used_prompt = prompt
        for text_prompt in build_prompt_candidates(prompt):
            inference_state = processor.set_text_prompt(
                state=inference_state, prompt=text_prompt
            )
            candidate_masks = inference_state.get("masks")
            if candidate_masks is not None and len(candidate_masks) > 0:
                masks = candidate_masks
                used_prompt = text_prompt
                break

        if masks is None or len(masks) == 0:
            print(f"  未检测到 {prompt}，原图直接输出")
            cv2.imwrite(output_path, image_cv)
            processor.reset_all_prompts(inference_state)
            _cleanup_gpu()
            return

        n_objects = len(masks)
        print(f"  检测到 {n_objects} 个 {prompt} (提示词: '{used_prompt}')")

        # 提取目标 box，用于动态调整网格密度
        boxes = inference_state.get("boxes")
        if boxes is not None and len(boxes) > 0:
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            box_sizes = [max(b[2] - b[0], b[3] - b[1]) for b in boxes]
        else:
            box_sizes = []
        object_scale = compute_face_scale(box_sizes, max(h, w))
        print(f"  scale={object_scale:.2f}")

        combined_mask = torch.any(masks.squeeze(1), dim=0).cpu().numpy()
        processor.reset_all_prompts(inference_state)

    if combined_mask.shape[:2] != (h, w):
        combined_mask = cv2.resize(
            combined_mask.astype(np.uint8), (w, h),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    adj_lw = max(1, int(line_width * object_scale))
    adj_ls = max(1, int(line_spacing * object_scale))
    grid_pattern = create_grid_pattern(h, w, adj_lw, adj_ls, angle)
    apply_region = combined_mask & grid_pattern
    frame_out = image_cv.copy()
    frame_out[apply_region] = 0

    cv2.imwrite(output_path, frame_out)
    print(f"  输出完成: {output_path}")
    _cleanup_gpu()


def create_app():
    import asyncio
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import FileResponse, JSONResponse
    from starlette.background import BackgroundTask

    app = FastAPI(title="人头/人体网格线处理 API")

    # 临时文件目录
    TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_temp")
    os.makedirs(TEMP_DIR, exist_ok=True)

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

    def validate_grid_params(line_width, line_spacing, angle):
        if not (0 <= angle <= 90):
            return "angle 必须在 0-90 之间"
        if line_width < 1:
            return "line_width 必须 >= 1"
        if line_spacing < 1:
            return "line_spacing 必须 >= 1"
        return None

    @app.post("/process/video")
    async def process_video_endpoint(
        file: UploadFile = File(..., description="上传视频文件"),
        prompt: str = Form("face", description="分割目标提示词，默认 face（如 person, glasses, truck, dog）"),
        line_width: int = Form(2, description="网格线宽度（像素）"),
        line_spacing: int = Form(9, description="网格线间距（像素）"),
        angle: float = Form(45.0, description="网格线角度（0-90 度）"),
    ):
        err = validate_grid_params(line_width, line_spacing, angle)
        if err:
            return JSONResponse(status_code=400, content={"error": err})

        effective_prompt = (prompt or "").strip() or "face"

        task_id = uuid.uuid4().hex[:12]
        ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
        input_path = os.path.join(TEMP_DIR, f"{task_id}_input{ext}")
        output_path = os.path.join(TEMP_DIR, f"{task_id}_output.mp4")

        try:
            content = await file.read()
            with open(input_path, "wb") as f:
                f.write(content)

            print(f"[{task_id}] 开始处理视频: {file.filename}, 目标: {effective_prompt}")
            await asyncio.to_thread(
                process_video, input_path, output_path, line_width, line_spacing, angle,
                effective_prompt
            )

            def cleanup():
                for p in (input_path, output_path):
                    if os.path.exists(p):
                        os.remove(p)

            return FileResponse(
                output_path,
                media_type="video/mp4",
                filename=f"grid_{file.filename or 'output.mp4'}",
                background=BackgroundTask(cleanup),
            )
        except ValueError as e:
            for p in (input_path, output_path):
                if os.path.exists(p):
                    os.remove(p)
            return JSONResponse(status_code=422, content={"error": str(e)})
        except Exception as e:
            for p in (input_path, output_path):
                if os.path.exists(p):
                    os.remove(p)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/process/image")
    async def process_image_endpoint(
        file: UploadFile = File(..., description="上传图片文件"),
        prompt: str = Form("face", description="分割目标提示词，默认 face（如 person, glasses, truck, dog）"),
        line_width: int = Form(2, description="网格线宽度（像素）"),
        line_spacing: int = Form(9, description="网格线间距（像素）"),
        angle: float = Form(45.0, description="网格线角度（0-90 度）"),
    ):
        err = validate_grid_params(line_width, line_spacing, angle)
        if err:
            return JSONResponse(status_code=400, content={"error": err})

        effective_prompt = (prompt or "").strip() or "face"

        task_id = uuid.uuid4().hex[:12]
        ext = os.path.splitext(file.filename or "image.png")[1] or ".png"
        input_path = os.path.join(TEMP_DIR, f"{task_id}_input{ext}")
        output_path = os.path.join(TEMP_DIR, f"{task_id}_output{ext}")

        # 根据扩展名确定 MIME 类型
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".bmp": "image/bmp",
            ".webp": "image/webp", ".tiff": "image/tiff", ".tif": "image/tiff",
        }
        media_type = mime_map.get(ext.lower(), "image/png")

        try:
            content = await file.read()
            with open(input_path, "wb") as f:
                f.write(content)

            print(f"[{task_id}] 开始处理图片: {file.filename}, 目标: {effective_prompt}")
            await asyncio.to_thread(
                process_image, input_path, output_path, line_width, line_spacing, angle,
                effective_prompt
            )

            def cleanup():
                for p in (input_path, output_path):
                    if os.path.exists(p):
                        os.remove(p)

            return FileResponse(
                output_path,
                media_type=media_type,
                filename=f"grid_{file.filename or 'output' + ext}",
                background=BackgroundTask(cleanup),
            )
        except ValueError as e:
            for p in (input_path, output_path):
                if os.path.exists(p):
                    os.remove(p)
            return JSONResponse(status_code=422, content={"error": str(e)})
        except Exception as e:
            for p in (input_path, output_path):
                if os.path.exists(p):
                    os.remove(p)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/process")
    async def process_auto(
        file: UploadFile = File(..., description="上传视频或图片文件（自动识别类型）"),
        prompt: str = Form("face", description="分割目标提示词，默认 face（如 person, glasses, truck, dog）"),
        line_width: int = Form(2, description="网格线宽度（像素）"),
        line_spacing: int = Form(9, description="网格线间距（像素）"),
        angle: float = Form(45.0, description="网格线角度（0-90 度）"),
    ):
        """自动根据文件扩展名判断是视频还是图片，分发到对应处理逻辑"""
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext in IMAGE_EXTS:
            return await process_image_endpoint(
                file, prompt, line_width, line_spacing, angle
            )
        elif ext in VIDEO_EXTS:
            return await process_video_endpoint(
                file, prompt, line_width, line_spacing, angle
            )
        else:
            return JSONResponse(
                status_code=400,
                content={"error": f"不支持的文件类型: {ext}，支持的格式: 图片{IMAGE_EXTS}, 视频{VIDEO_EXTS}"}
            )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "video_model_loaded": _predictor is not None,
            "image_model_loaded": _image_processor is not None,
        }

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="人脸/眼镜网格线视频处理 API 服务")
    parser.add_argument("--host", default="127.0.0.1",
                        help="监听地址，默认 127.0.0.1（仅本机）。"
                             "设为 0.0.0.0 会绑定所有网卡，服务无鉴权，请仅在可信网络使用")
    parser.add_argument("--port", type=int, default=8000, help="监听端口，默认 8000")
    parser.add_argument("--version", choices=["3.0", "3.1"], default="3.0",
                        help="SAM3 模型版本，默认 3.0")
    parser.add_argument("--checkpoint", default=None, help="手动指定模型权重路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    checkpoint = args.checkpoint or CHECKPOINTS[args.version]

    # 启动时加载视频模型
    _predictor = build_predictor(checkpoint, version=args.version)

    # mmgp 卸载视频模型组件（对齐 inference.py _try_apply_mmgp_to_video_predictor）
    _vmodel = getattr(_predictor, "model", None)
    if _vmodel is not None:
        # 在 mmgp 卸载前强制触发 _device 属性缓存为 CUDA
        for _obj in [_vmodel, getattr(_vmodel, "detector", None)]:
            if _obj is not None:
                try:
                    if isinstance(getattr(type(_obj), "device", None), property):
                        _cached = _obj.device
                        if _cached is not None and str(_cached) != "cpu":
                            _obj._device = _cached
                except Exception:
                    pass
        # 1. detector.backbone
        _detector = getattr(_vmodel, "detector", None)
        _bb = getattr(_detector, "backbone", None) if _detector is not None else None
        if _bb is not None:
            _apply_mmgp(
                _bb, "video.detector.backbone",
                module_key="transformer",
                quantizeTransformer=False, asyncTransfers=False,
            )
        # 2. tracker
        _tracker = getattr(_vmodel, "tracker", None)
        if _tracker is not None:
            _apply_mmgp(
                _tracker, "video.tracker",
                module_key="transformer",
                quantizeTransformer=False, asyncTransfers=False,
            )
        # 3. SAM3.1 专项：tracker 按帧卸载 + 批量控制
        if args.version == "3.1":
            _inner = getattr(_tracker, "model", None) if _tracker is not None else None
            if _inner is not None and hasattr(_inner, "offload_output_to_cpu_for_eval"):
                _inner.offload_output_to_cpu_for_eval = True
                print("[MMGP] SAM3.1 tracker: offload_output_to_cpu_for_eval=True")
            if _inner is not None and hasattr(_inner, "trim_past_non_cond_mem_for_eval"):
                _inner.trim_past_non_cond_mem_for_eval = True
                print("[MMGP] SAM3.1 tracker: trim_past_non_cond_mem_for_eval=True")
            if hasattr(_vmodel, "use_batched_grounding"):
                _vmodel.use_batched_grounding = False
            if hasattr(_vmodel, "postprocess_batch_size"):
                _vmodel.postprocess_batch_size = 1
                print("[MMGP] SAM3.1: postprocess_batch_size=1")

    # 启动时加载图片模型
    print("正在加载图片模型...")
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    with torch.autocast("cuda", dtype=torch.bfloat16):
        _image_model = build_sam3_image_model(
            checkpoint_path=checkpoint,
            bpe_path=BPE_PATH,
        )
    _apply_mmgp(
        _image_model, "image.model",
        quantizeTransformer=False, asyncTransfers=False,
        pinnedMemory=True, budgets=None,
    )
    # mmgp 将 Linear 权重存为 BFloat16，但 decoder.py forward_ffn 禁用 autocast
    # 导致 LayerNorm 输出 float32，注册 hook 将输入自动对齐到权重 dtype
    def _cast_to_weight_dtype(module, args):
        if module.weight is not None and len(args) > 0:
            x = args[0]
            if isinstance(x, torch.Tensor) and x.dtype != module.weight.dtype:
                return (x.to(dtype=module.weight.dtype),) + args[1:]
        return args
    for _submod in _image_model.modules():
        if isinstance(_submod, torch.nn.Linear):
            _submod.register_forward_pre_hook(_cast_to_weight_dtype)
    _image_processor = Sam3Processor(_image_model, confidence_threshold=0.5)
    print("图片模型加载完成")

    app = create_app()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
