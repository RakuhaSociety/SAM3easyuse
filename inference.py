"""
SAM3 Inference — 统一推理接口 & CLI 工具

用法 (CLI):
    # 文本分割
    python inference.py image-text -i photo.jpg -t "person, car" -o result.png

    # 框选分割
    python inference.py image-box -i photo.jpg --box 100,50,400,300 -o result.png

    # 点击分割 (交互式)
    python inference.py image-points -i photo.jpg --points 200,150,1 350,200,0 -o result.png

    # 批量分割（文件夹）
    python inference.py batch -d ./images -t "person" -o ./results

    # 批量分割（视频拆帧）
    python inference.py batch -v input.mp4 -t "car" --interval 5 -o ./results

    # 视频文本跟踪
    python inference.py video-text -v input.mp4 -t "person" -o tracked.mp4

    # 视频点击跟踪
    python inference.py video-points -v input.mp4 --points 200,150,1 --frame 30 -o tracked.mp4

    # 视频框选跟踪（多框 + 正/负向）
    python inference.py video-box -v input.mp4 --box 100,50,400,300 -o tracked.mp4

    # 视频框选 + 文本跟踪
    python inference.py video-box -v input.mp4 --box 100,50,400,300 -t "person" -o tracked.mp4

    # 多个正向框 + 负向框组合
    python inference.py video-box -v input.mp4 --box 100,50,400,300 --neg-box 50,50,80,80 -t "face" -o tracked.mp4

作为库使用:
    from inference import SAM3Inference

    sam = SAM3Inference(version="sam3.1", use_fa=True)
    result, info = sam.segment_image_text("photo.jpg", "person, car")
    result, info = sam.track_video_text("input.mp4", "person")
"""

import argparse
import gc
import os
import subprocess
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from mmgp import offload as _mmgp_offload
    from mmgp import profile_type as _mmgp_profile_type

    _MMGP_AVAILABLE = True
except Exception:
    _mmgp_offload = None
    _mmgp_profile_type = None
    _MMGP_AVAILABLE = False

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "sam3"))
SAM3_DIR = os.path.join(BASE_DIR, "sam3")
CHECKPOINT_SAM3 = os.path.join(SAM3_DIR, "checkpoints", "sam3.pt")
CHECKPOINT_SAM31 = os.path.join(SAM3_DIR, "checkpoints", "sam3.1_multiplex.pt")
BPE_PATH = os.path.join(SAM3_DIR, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
    (60, 100, 170), (200, 80, 120), (100, 200, 100), (180, 120, 60),
]


# ===================================================================
# 渲染工具
# ===================================================================

def masks_to_binary(masks, h: int, w: int) -> np.ndarray:
    """多个 mask → 合并二值图 (H×W×3, uint8, 白=物体)"""
    combined = np.zeros((h, w), dtype=np.uint8)
    n = len(masks) if isinstance(masks, list) else (masks.shape[0] if hasattr(masks, "shape") else 0)
    for i in range(n):
        m = masks[i]
        if isinstance(m, torch.Tensor):
            m = m.cpu().numpy()
        m = m.squeeze()
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        combined[m > 0.5] = 255
    return np.stack([combined] * 3, axis=-1)


def overlay_masks(image_np: np.ndarray, masks, boxes, scores, alpha: float = 0.45) -> np.ndarray:
    """在图像上叠加 mask + 框 + 分数标签"""
    overlay = image_np.copy()
    h, w = overlay.shape[:2]
    n = scores.shape[0] if isinstance(scores, torch.Tensor) else len(scores)
    for i in range(n):
        color = COLORS[i % len(COLORS)]
        m = masks[i]
        if isinstance(m, torch.Tensor):
            m = m.cpu().numpy()
        m = m.squeeze()
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bool = m > 0.5
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_bool,
                np.clip(alpha * color[c] + (1 - alpha) * overlay[:, :, c], 0, 255).astype(np.uint8),
                overlay[:, :, c],
            )
        box = boxes[i]
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        x0, y0, x1, y1 = box.astype(int)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
        score = scores[i].item() if isinstance(scores[i], torch.Tensor) else float(scores[i])
        label = f"#{i} {score:.2f}"
        (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay, (x0, max(y0 - th_t - 8, 0)), (x0 + tw + 4, y0), color, -1)
        cv2.putText(overlay, label, (x0 + 2, max(y0 - 4, th_t + 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def overlay_video_masks(frame_rgb: np.ndarray, outputs: dict, alpha: float = 0.45) -> np.ndarray:
    """视频跟踪输出 → 叠加渲染"""
    overlay = frame_rgb.copy()
    h, w = overlay.shape[:2]
    masks = outputs.get("out_binary_masks", [])
    obj_ids = outputs.get("out_obj_ids", [])
    boxes = outputs.get("out_boxes_xywh", [])
    probs = outputs.get("out_probs", [])
    n = len(obj_ids) if hasattr(obj_ids, "__len__") else (
        obj_ids.shape[0] if isinstance(obj_ids, (torch.Tensor, np.ndarray)) else 0)
    for i in range(n):
        oid = int(obj_ids[i].item() if isinstance(obj_ids[i], torch.Tensor) else obj_ids[i])
        color = COLORS[oid % len(COLORS)]
        m = masks[i]
        if isinstance(m, torch.Tensor):
            m = m.cpu().numpy()
        m = m.squeeze()
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bool = m > 0.5
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_bool,
                np.clip(alpha * color[c] + (1 - alpha) * overlay[:, :, c], 0, 255).astype(np.uint8),
                overlay[:, :, c],
            )
        if i < len(boxes):
            box = boxes[i]
            if isinstance(box, torch.Tensor):
                box = box.cpu().tolist()
            elif isinstance(box, np.ndarray):
                box = box.tolist()
            bx, by, bw, bh = box
            x1, y1 = int(bx * w), int(by * h)
            x2, y2 = int((bx + bw) * w), int((by + bh) * h)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            prob = probs[i] if i < len(probs) else None
            if isinstance(prob, torch.Tensor):
                prob = prob.item()
            lbl = f"id={oid}" + (f" {prob:.2f}" if prob is not None else "")
            (tw, th_t), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x1, max(y1 - th_t - 6, 0)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(overlay, lbl, (x1 + 2, max(y1 - 3, th_t + 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def video_masks_to_binary(outputs: dict, h: int, w: int) -> np.ndarray:
    """视频跟踪输出 → 二值帧"""
    masks = outputs.get("out_binary_masks", [])
    n = len(masks) if isinstance(masks, list) else (masks.shape[0] if hasattr(masks, "shape") else 0)
    combined = np.zeros((h, w), dtype=np.uint8)
    for i in range(n):
        m = masks[i]
        if isinstance(m, torch.Tensor):
            m = m.cpu().numpy()
        m = m.squeeze()
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        combined[m > 0.5] = 255
    return np.stack([combined] * 3, axis=-1)


def _write_video(frames_rgb: list, fps: float, output_path: str):
    """RGB 帧列表 → H.264 MP4"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    h, w = frames_rgb[0].shape[:2]
    temp_path = output_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))
    for f in frames_rgb:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_path,
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-pix_fmt", "yuv420p", output_path],
            check=True, capture_output=True,
        )
        os.remove(temp_path)
    except Exception:
        if os.path.exists(temp_path):
            os.replace(temp_path, output_path)


def _read_video_frames(video_path: str) -> Tuple[list, float]:
    """读取全部帧 (RGB np.ndarray list) 和 fps"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"无法读取视频帧: {video_path}")
    return frames, fps


# ===================================================================
# SAM3Inference — 主推理类
# ===================================================================

class SAM3Inference:
    """SAM3/SAM3.1 统一推理接口

    示例::

        sam = SAM3Inference(version="sam3.1", use_fa=True)

        # 图片 — 文本分割
        result_img, info = sam.segment_image_text("photo.jpg", "person, car")

        # 图片 — 框选分割
        result_img, info = sam.segment_image_box("photo.jpg", boxes=[(100,50,400,300)])

        # 图片 — 点分割
        result_img, info = sam.segment_image_points("photo.jpg", points=[(200,150,1)])

        # 批量分割（文件夹）
        results, info = sam.batch_segment_folder("./images", "person")

        # 批量分割（视频拆帧）
        results, video_path, info = sam.batch_segment_video("input.mp4", "car", interval=5)

        # 视频 — 文本跟踪
        output_path, info = sam.track_video_text("input.mp4", "person")

        # 视频 — 点击跟踪
        output_path, info = sam.track_video_points("input.mp4", points=[(200,150,1)], frame_idx=30)

        # 视频 — 框选跟踪 (± 文本)
        output_path, info = sam.track_video_box("input.mp4", box=(100,50,400,300), text="person")
    """

    def __init__(
        self,
        version: str = "sam3",
        use_fa: bool = True,
        confidence: float = 0.5,
        mask_mode: bool = False,
        use_mmgp: bool = False,
        mmgp_profile: int = 4,
        sam31_batch_size: int = 16,
        output_dir: str = OUTPUT_DIR,
        checkpoint_sam3: str = CHECKPOINT_SAM3,
        checkpoint_sam31: str = CHECKPOINT_SAM31,
        bpe_path: str = BPE_PATH,
    ):
        """
        Args:
            version: "sam3" 或 "sam3.1"
            use_fa: 是否使用 Flash Attention 2（仅视频模型）
            confidence: 文本检测置信度阈值
            mask_mode: True=输出二值 mask, False=叠加可视化
            output_dir: 输出目录
            checkpoint_sam3: SAM3 检查点路径
            checkpoint_sam31: SAM3.1 检查点路径
            bpe_path: BPE 词表路径
        """
        self.version = version
        self.use_fa = use_fa
        self.confidence = confidence
        self.mask_mode = mask_mode
        self.use_mmgp = use_mmgp
        self.mmgp_profile = int(mmgp_profile)
        self.sam31_batch_size = int(sam31_batch_size)
        self.output_dir = output_dir
        self.checkpoint_sam3 = checkpoint_sam3
        self.checkpoint_sam31 = checkpoint_sam31
        self.bpe_path = bpe_path

        self._image_processor = None
        self._interactive_model = None
        self._interactive_processor = None
        self._video_predictor = None
        self._video_predictor_fa = None  # 记录构建时的 FA 设置
        self._active_mode = None  # "image" / "interactive" / "video"
        # target_name -> offload_obj：mmgp profile() 的返回对象，用于后续 release()
        # 释放 pinned host memory（Windows 任务管理器「共享 GPU」上的占用）
        self._mmgp_applied: Dict[str, object] = {}

        if self.use_mmgp and not _MMGP_AVAILABLE:
            print("[MMGP] 警告: 未检测到 mmgp，已忽略 --mmgp 参数")

    @property
    def _ckpt(self) -> str:
        return self.checkpoint_sam31 if self.version == "sam3.1" else self.checkpoint_sam3

    # ------ 资源管理 ------

    def _cleanup_gpu(self):
        """释放 GPU 缓存与 pinned host memory（共享 GPU 显存）。

        torch.cuda.empty_cache() 只清 CUDA 设备缓存，不清 pinned host
        memory；后者要靠 torch._C._host_emptyCache()，Windows 还需要
        EmptyWorkingSet() 才能让任务管理器「共享 GPU」数字真正下降。
        优先调用 mmgp.offload.flush_torch_caches() 一站式处理。
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if _MMGP_AVAILABLE:
            try:
                _mmgp_offload.flush_torch_caches()
                return
            except Exception as e:
                print(f"[MMGP] flush_torch_caches 失败: {e}")

        # mmgp 不可用时的回退
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

    def _release_mmgp_for(self, prefix: str):
        """释放 target_name 以 prefix 开头的所有 mmgp offload 对象，
        归还 pinned host memory。"""
        if not self._mmgp_applied:
            return
        keys = [k for k in list(self._mmgp_applied.keys()) if k.startswith(prefix)]
        for k in keys:
            obj = self._mmgp_applied.pop(k, None)
            if obj is None:
                continue
            try:
                release_fn = getattr(obj, "release", None)
                if callable(release_fn):
                    release_fn()
                    print(f"[MMGP] 已释放 offload ({k})")
            except Exception as e:
                print(f"[MMGP] 释放 offload 失败 ({k}): {e}")

    def _unload(self, mode: str):
        if mode == "image" and self._image_processor is not None:
            del self._image_processor
            self._image_processor = None
            self._release_mmgp_for("image.")
        elif mode == "interactive" and self._interactive_model is not None:
            del self._interactive_model, self._interactive_processor
            self._interactive_model = None
            self._interactive_processor = None
            self._release_mmgp_for("interactive.")
        elif mode == "video" and self._video_predictor is not None:
            del self._video_predictor
            self._video_predictor = None
            self._video_predictor_fa = None
            self._release_mmgp_for("video.")

    def _ensure_mode(self, mode: str):
        """切换模式，卸载其他模型以节省显存"""
        if self._active_mode == mode:
            return
        for m in ("image", "interactive", "video"):
            if m != mode:
                self._unload(m)
        self._cleanup_gpu()
        self._active_mode = mode

    def unload_all(self):
        """手动释放所有模型和显存"""
        for m in ("image", "interactive", "video"):
            self._unload(m)
        self._cleanup_gpu()
        self._active_mode = None

    # ------ 模型加载 ------

    def _get_image_processor(self):
        if self._image_processor is None:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            print(f"[SAM3] 加载图像分割模型 ({self.version})...")
            # 与 gradio 一致：始终在 GPU 加载，代 mmgp 后续按需搬运。
            # pinnedMemory=True 强制全部权重使用 pinned host memory，任务管理器的
            # 「共享 GPU」会上涨；budgets=None 避免默认 budgets["transformer"]=1200
            # 把子模块滑留 CPU（predict_inst 会绕过顶层 forward 直接调用
            # sam_prompt_encoder，导致 mmgp hook 不触发）。
            model = build_sam3_image_model(
                bpe_path=self.bpe_path, checkpoint_path=self._ckpt,
                device=DEVICE,
            )
            self._apply_mmgp(
                model, f"image.{self.version}.model",
                quantizeTransformer=False, asyncTransfers=False,
                pinnedMemory=True, budgets=None,
            )
            if self.use_mmgp and _MMGP_AVAILABLE:
                # decoder.py forward_ffn 显式禁用了 autocast，导致 LayerNorm 输出 float32
                # 而 mmgp 存储/恢复的 Linear 权重为 BFloat16，类型不匹配
                # 注册 forward pre-hook 将 Linear 输入自动对齐到权重 dtype
                def _cast_to_weight_dtype(module, args):
                    if module.weight is not None and len(args) > 0:
                        x = args[0]
                        if isinstance(x, torch.Tensor) and x.dtype != module.weight.dtype:
                            return (x.to(dtype=module.weight.dtype),) + args[1:]
                    return args
                for submod in model.modules():
                    if isinstance(submod, torch.nn.Linear):
                        submod.register_forward_pre_hook(_cast_to_weight_dtype)
            self._image_processor = Sam3Processor(model, confidence_threshold=self.confidence)
            print("[SAM3] 图像分割模型加载完成 [OK]")
        self._image_processor.confidence_threshold = self.confidence
        return self._image_processor

    def _get_interactive(self):
        if self._interactive_model is None:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            print(f"[SAM3] 加载交互式分割模型 ({self.version})...")
            # 交互式点击分割仅 sam3 支持（sam3.1_multiplex.pt 不含相关权重）
            ckpt = self.checkpoint_sam3
            self._interactive_model = build_sam3_image_model(
                bpe_path=self.bpe_path, checkpoint_path=ckpt,
                enable_inst_interactivity=True,
            )
            self._apply_mmgp(
                self._interactive_model, "interactive.sam3.model",
                quantizeTransformer=False, asyncTransfers=False,
                pinnedMemory=True, budgets=None,
            )
            self._interactive_processor = Sam3Processor(self._interactive_model)
            print("[SAM3] 交互式分割模型加载完成 [OK]")
        return self._interactive_model, self._interactive_processor

    def _get_video_predictor(self):
        if self._video_predictor is not None and self._video_predictor_fa != self.use_fa:
            del self._video_predictor
            self._video_predictor = None
            self._release_mmgp_for("video.")
            self._cleanup_gpu()
        if self._video_predictor is None:
            from sam3.model_builder import build_sam3_predictor
            fa_label = "FA2" if self.use_fa else "SDPA"
            print(f"[SAM3] 加载视频模型 ({self.version}, {fa_label})...")
            self._video_predictor = build_sam3_predictor(
                version=self.version,
                checkpoint_path=self._ckpt,
                bpe_path=self.bpe_path,
                use_fa3=self.use_fa,
            )
            self._try_apply_mmgp_to_video_predictor(self._video_predictor)
            self._video_predictor_fa = self.use_fa
            print("[SAM3] 视频模型加载完成 [OK]")
        return self._video_predictor

    def _get_tracker(self):
        predictor = self._get_video_predictor()
        model = predictor.model
        tracker = model.tracker
        if hasattr(model, "detector") and hasattr(model.detector, "backbone"):
            tracker.backbone = model.detector.backbone
        return tracker

    def _get_mmgp_profile(self):
        if not _MMGP_AVAILABLE:
            return None

        profile_names = {
            1: "HighRAM_HighVRAM",
            2: "HighRAM_LowVRAM",
            3: "LowRAM_HighVRAM",
            4: "LowRAM_LowVRAM",
        }

        preferred = [
            profile_names.get(int(self.mmgp_profile)),
            "LowRAM_LowVRAM",
            "HighRAM_LowVRAM",
            "LowRAM_HighVRAM",
            "HighRAM_HighVRAM",
        ]
        for name in preferred:
            if name and hasattr(_mmgp_profile_type, name):
                return getattr(_mmgp_profile_type, name)

        for name in dir(_mmgp_profile_type):
            if name.startswith("_"):
                continue
            return getattr(_mmgp_profile_type, name)
        return None

    def _apply_mmgp(self, module, target_name: str, module_key: str = "model", **override_kwargs):
        if not self.use_mmgp:
            return
        if not _MMGP_AVAILABLE:
            return
        if module is None:
            return

        key = target_name
        if key in self._mmgp_applied:
            return

        profile = self._get_mmgp_profile()
        if profile is None:
            print("[MMGP] 未找到可用 profile，跳过")
            return

        offload_obj = None
        try:
            offload_obj = _mmgp_offload.profile(module, profile_no=profile, **override_kwargs)
        except Exception as e1:
            try:
                offload_obj = _mmgp_offload.profile({module_key: module}, profile_no=profile, **override_kwargs)
            except Exception as e2:
                print(f"[MMGP] 应用失败 ({target_name}): {e2} (direct={e1})")
                return

        self._mmgp_applied[key] = offload_obj
        try:
            if hasattr(torch, "set_default_device"):
                torch.set_default_device("cpu")
        except Exception:
            pass
        if override_kwargs:
            print(f"[MMGP] 已应用到 {target_name} (profile={self.mmgp_profile}, overrides={override_kwargs})")
        else:
            print(f"[MMGP] 已应用到 {target_name} (profile={self.mmgp_profile})")

    def _try_apply_mmgp_to_video_predictor(self, predictor):
        if not self.use_mmgp:
            return
        model = getattr(predictor, "model", None)
        if model is None:
            return

        # 在 mmgp 卸载参数前，强制触发 _device 缓存为 CUDA。
        # sam3_video_base.py 的 device 属性会 lazy 地缓存 next(parameters()).device；
        # 卸载后第一次取值若不缓存，会拿到 CPU param 的 device，导致 image 被移到 CPU。
        for _obj in [model, getattr(model, "detector", None)]:
            if _obj is not None:
                try:
                    # hasattr(type, "device") 比 callable(property) 更可靠
                    if isinstance(getattr(type(_obj), "device", None), property):
                        _cached = _obj.device
                        if _cached is not None and str(_cached) != "cpu":
                            _obj._device = _cached
                except Exception:
                    pass

        # 1. 挂载 detector.backbone（vision_backbone ~1.76GB + language_backbone ~1.35GB）
        #    text_encoder_ve.py 中已修复 token_embedding 设备错配问题，因此可安全卸载。
        _detector = getattr(model, "detector", None)
        _bb = getattr(_detector, "backbone", None) if _detector is not None else None
        if _bb is not None:
            self._apply_mmgp(
                _bb,
                f"video.{self.version}.detector.backbone",
                module_key="transformer",
                quantizeTransformer=False,
                asyncTransfers=False,
            )

        # 2. 挂载 tracker（~45MB）
        tracker = getattr(model, "tracker", None)
        if tracker is not None:
            self._apply_mmgp(
                tracker,
                f"video.{self.version}.tracker",
                module_key="transformer",
                quantizeTransformer=False,
                asyncTransfers=False,
            )

        # 3. SAM3.1 专项：把 tracker 推理状态按帧卸载到 CPU RAM
        if self.version == "sam3.1":
            _inner = getattr(tracker, "model", None) if tracker is not None else None
            if _inner is not None and hasattr(_inner, "offload_output_to_cpu_for_eval"):
                _inner.offload_output_to_cpu_for_eval = True
                print("[MMGP] SAM3.1 tracker: 已启用 offload_output_to_cpu_for_eval=True")
            if _inner is not None and hasattr(_inner, "trim_past_non_cond_mem_for_eval"):
                _inner.trim_past_non_cond_mem_for_eval = True
                print("[MMGP] SAM3.1 tracker: 已启用 trim_past_non_cond_mem_for_eval=True")

        # 4. SAM3.1 专项：控制批量 grounding 大小以平衡显存与速度
        if self.version == "sam3.1" and model is not None:
            _eff_batch = self.sam31_batch_size
            _use_batched = _eff_batch > 1
            if hasattr(model, "use_batched_grounding"):
                model.use_batched_grounding = _use_batched
                print(f"[MMGP] SAM3.1: use_batched_grounding={_use_batched} (batch={_eff_batch})")
            if hasattr(model, "batched_grounding_batch_size"):
                model.batched_grounding_batch_size = _eff_batch
            if hasattr(model, "postprocess_batch_size"):
                model.postprocess_batch_size = _eff_batch
                print(f"[MMGP] SAM3.1: postprocess_batch_size={_eff_batch}")


    # ==================================================================
    # 图片 — 文本分割
    # ==================================================================

    def segment_image_text(
        self,
        image: Union[str, np.ndarray, Image.Image],
        text: str,
        confidence: Optional[float] = None,
        mask_mode: Optional[bool] = None,
    ) -> Tuple[np.ndarray, dict]:
        """使用文本提示分割图像

        Args:
            image: 图片路径、numpy 数组或 PIL Image
            text: 文本提示，如 "person, car"
            confidence: 置信度 (覆盖实例默认)
            mask_mode: 输出二值 mask (覆盖实例默认)

        Returns:
            (result_image, info_dict)
            info_dict 包含 n_objects, scores, elapsed 等
        """
        self._ensure_mode("image")
        conf = confidence if confidence is not None else self.confidence
        mmode = mask_mode if mask_mode is not None else self.mask_mode

        pil = self._to_pil(image)
        processor = self._get_image_processor()
        processor.confidence_threshold = conf

        t0 = time.time()
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            state = processor.set_image(pil)
            state = processor.set_text_prompt(state=state, prompt=text.strip())
        elapsed = time.time() - t0

        masks, boxes, scores = state["masks"], state["boxes"], state["scores"]
        n = scores.shape[0] if isinstance(scores, torch.Tensor) else len(scores)

        img_np = np.array(pil)
        if n == 0:
            result = np.zeros_like(img_np) if mmode else img_np
        elif mmode:
            result = masks_to_binary(masks, *img_np.shape[:2])
        else:
            result = overlay_masks(img_np, masks, boxes, scores)

        score_list = [float(scores[i].item() if isinstance(scores[i], torch.Tensor) else scores[i]) for i in range(n)]
        self._cleanup_gpu()
        return result, {"n_objects": n, "scores": score_list, "elapsed": elapsed}

    # ==================================================================
    # 图片 — 框选分割
    # ==================================================================

    def segment_image_box(
        self,
        image: Union[str, np.ndarray, Image.Image],
        boxes: List[Tuple[int, int, int, int]],
        text: str = "",
        confidence: Optional[float] = None,
        mask_mode: Optional[bool] = None,
    ) -> Tuple[np.ndarray, dict]:
        """使用框选提示分割图像，可选结合文本

        Args:
            image: 图片
            boxes: [(x1,y1,x2,y2), ...] 像素坐标，默认正向框
            text: 可选文本提示
            confidence: 置信度
            mask_mode: 二值 mask

        Returns:
            (result_image, info_dict)
        """
        self._ensure_mode("image")
        conf = confidence if confidence is not None else self.confidence
        mmode = mask_mode if mask_mode is not None else self.mask_mode

        pil = self._to_pil(image)
        w_img, h_img = pil.size
        processor = self._get_image_processor()
        processor.confidence_threshold = conf

        t0 = time.time()
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            state = processor.set_image(pil)
            processor.reset_all_prompts(state)
            text = (text or "").strip()
            if text:
                state = processor.set_text_prompt(state=state, prompt=text)
            for bx in boxes:
                x1, y1, x2, y2 = bx[:4]
                is_pos = bx[4] if len(bx) > 4 else True
                cx = (x1 + x2) / 2.0 / w_img
                cy = (y1 + y2) / 2.0 / h_img
                bw = (x2 - x1) / w_img
                bh = (y2 - y1) / h_img
                state = processor.add_geometric_prompt(
                    state=state, box=[cx, cy, bw, bh], label=is_pos,
                )
        elapsed = time.time() - t0

        masks, boxes_out, scores = state["masks"], state["boxes"], state["scores"]
        n = scores.shape[0] if isinstance(scores, torch.Tensor) else len(scores)

        # 过滤：中心在正向框内
        pos_boxes = [(bx[0], bx[1], bx[2], bx[3]) for bx in boxes if (bx[4] if len(bx) > 4 else True)]
        if n > 0 and pos_boxes:
            keep = []
            for i in range(n):
                det = boxes_out[i]
                if isinstance(det, torch.Tensor):
                    det = det.cpu().numpy()
                dx0, dy0, dx1, dy1 = det.flatten()[:4]
                dcx, dcy = (dx0 + dx1) / 2, (dy0 + dy1) / 2
                for ux1, uy1, ux2, uy2 in pos_boxes:
                    if ux1 <= dcx <= ux2 and uy1 <= dcy <= uy2:
                        keep.append(i)
                        break
            if keep:
                masks, boxes_out, scores = masks[keep], boxes_out[keep], scores[keep]
                n = len(keep)

        img_np = np.array(pil)
        if n == 0:
            result = np.zeros_like(img_np) if mmode else img_np
        elif mmode:
            result = masks_to_binary(masks, h_img, w_img)
        else:
            result = overlay_masks(img_np, masks, boxes_out, scores)

        score_list = [float(scores[i].item() if isinstance(scores[i], torch.Tensor) else scores[i]) for i in range(n)]
        self._cleanup_gpu()
        return result, {"n_objects": n, "scores": score_list, "elapsed": elapsed}

    # ==================================================================
    # 图片 — 点击分割
    # ==================================================================

    def segment_image_points(
        self,
        image: Union[str, np.ndarray, Image.Image],
        points: List[Tuple[int, int, int]],
        mask_mode: Optional[bool] = None,
    ) -> Tuple[np.ndarray, dict]:
        """使用点提示分割图像

        Args:
            image: 图片
            points: [(x, y, label), ...] label=1 前景, 0 背景, 像素坐标
            mask_mode: 二值 mask

        Returns:
            (result_image, info_dict)  info_dict 含 best_score, all_scores
        """
        self._ensure_mode("interactive")
        mmode = mask_mode if mask_mode is not None else self.mask_mode

        pil = self._to_pil(image)
        model, processor = self._get_interactive()

        point_coords = np.array([[x, y] for x, y, _ in points])
        point_labels = np.array([l for _, _, l in points])

        t0 = time.time()
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            state = processor.set_image(pil)
            masks, scores, logits = model.predict_inst(
                state,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
        elapsed = time.time() - t0

        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = float(scores[best_idx])

        img_np = np.array(pil)
        h, w = img_np.shape[:2]
        if mmode:
            binary = np.zeros((h, w), dtype=np.uint8)
            binary[best_mask > 0.5] = 255
            result = np.stack([binary] * 3, axis=-1)
        else:
            overlay = img_np.copy()
            color = COLORS[0]
            mask_bool = best_mask > 0.5
            alpha = 0.45
            for c in range(3):
                overlay[:, :, c] = np.where(
                    mask_bool,
                    np.clip(alpha * color[c] + (1 - alpha) * overlay[:, :, c], 0, 255).astype(np.uint8),
                    overlay[:, :, c],
                )
            contours, _ = cv2.findContours(
                mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, color, 2)
            result = overlay

        self._cleanup_gpu()
        return result, {
            "best_score": best_score,
            "all_scores": [float(s) for s in scores],
            "elapsed": elapsed,
        }

    # ==================================================================
    # 批量分割 — 文件夹
    # ==================================================================

    def batch_segment_folder(
        self,
        folder: str,
        text: str,
        confidence: Optional[float] = None,
        mask_mode: Optional[bool] = None,
        output_dir: Optional[str] = None,
        callback=None,
    ) -> Tuple[List[Tuple[str, int]], dict]:
        """批量分割文件夹中的图片

        Args:
            folder: 图片文件夹路径
            text: 文本提示
            confidence: 置信度
            mask_mode: 二值 mask
            output_dir: 结果输出目录 (默认 self.output_dir/batch_xxx)
            callback: 可选进度回调 callback(current, total, filename)

        Returns:
            (results, info_dict)
            results: [(save_path, n_objects), ...]
            info_dict: total, elapsed, avg_time, output_dir
        """
        self._ensure_mode("image")
        conf = confidence if confidence is not None else self.confidence
        mmode = mask_mode if mask_mode is not None else self.mask_mode

        folder = folder.strip().strip('"').strip("'")
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"文件夹不存在: {folder}")

        files = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ])
        if not files:
            raise FileNotFoundError(f"文件夹中未找到图片: {folder}")

        out_dir = output_dir or os.path.join(self.output_dir, f"batch_{int(time.time())}")
        os.makedirs(out_dir, exist_ok=True)

        processor = self._get_image_processor()
        processor.confidence_threshold = conf

        results = []
        t0 = time.time()
        for idx, fpath in enumerate(files):
            name = os.path.basename(fpath)
            if callback:
                callback(idx, len(files), name)
            pil = Image.open(fpath).convert("RGB")
            with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
                state = processor.set_image(pil)
                state = processor.set_text_prompt(state=state, prompt=text.strip())

            masks, boxes, scores = state["masks"], state["boxes"], state["scores"]
            n = scores.shape[0] if isinstance(scores, torch.Tensor) else len(scores)
            img_np = np.array(pil)
            if n > 0:
                res = masks_to_binary(masks, *img_np.shape[:2]) if mmode else overlay_masks(img_np, masks, boxes, scores)
            else:
                res = np.zeros_like(img_np) if mmode else img_np

            save_path = os.path.join(out_dir, f"{os.path.splitext(name)[0]}.png")
            Image.fromarray(res).save(save_path)
            results.append((save_path, n))

        elapsed = time.time() - t0
        self._cleanup_gpu()
        return results, {
            "total": len(files),
            "elapsed": elapsed,
            "avg_time": elapsed / len(files) if files else 0,
            "output_dir": out_dir,
        }

    # ==================================================================
    # 批量分割 — 视频拆帧
    # ==================================================================

    def batch_segment_video(
        self,
        video_path: str,
        text: str,
        interval: int = 1,
        confidence: Optional[float] = None,
        mask_mode: Optional[bool] = None,
        output_dir: Optional[str] = None,
        callback=None,
    ) -> Tuple[List[Tuple[str, int]], Optional[str], dict]:
        """批量分割视频帧

        Args:
            video_path: 视频路径
            text: 文本提示
            interval: 抽帧间隔 (每 N 帧取 1 帧)
            confidence: 置信度
            mask_mode: 二值 mask
            output_dir: 输出目录
            callback: 进度回调 callback(current, total, label)

        Returns:
            (results, output_video_path, info_dict)
        """
        self._ensure_mode("image")
        conf = confidence if confidence is not None else self.confidence
        mmode = mask_mode if mask_mode is not None else self.mask_mode
        interval = max(1, interval)

        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        all_frames = []
        fidx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fidx % interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_frames.append((fidx, Image.fromarray(rgb)))
            fidx += 1
        cap.release()

        if not all_frames:
            raise ValueError("视频中未读取到任何帧")

        out_dir = output_dir or os.path.join(self.output_dir, f"batch_{int(time.time())}")
        os.makedirs(out_dir, exist_ok=True)

        processor = self._get_image_processor()
        processor.confidence_threshold = conf

        results = []
        result_frames = []
        t0 = time.time()
        total = len(all_frames)
        for i, (frame_idx, pil_img) in enumerate(all_frames):
            label = f"frame_{frame_idx:06d}"
            if callback:
                callback(i, total, label)
            with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
                state = processor.set_image(pil_img)
                state = processor.set_text_prompt(state=state, prompt=text.strip())

            masks, boxes, scores = state["masks"], state["boxes"], state["scores"]
            n = scores.shape[0] if isinstance(scores, torch.Tensor) else len(scores)
            img_np = np.array(pil_img)
            if n > 0:
                res = masks_to_binary(masks, *img_np.shape[:2]) if mmode else overlay_masks(img_np, masks, boxes, scores)
            else:
                res = np.zeros_like(img_np) if mmode else img_np

            save_path = os.path.join(out_dir, f"{label}.png")
            Image.fromarray(res).save(save_path)
            results.append((save_path, n))
            result_frames.append(res)

        elapsed = time.time() - t0

        # 拼回视频
        out_fps = video_fps / interval
        video_out = os.path.join(out_dir, "result_video.mp4")
        _write_video(result_frames, out_fps, video_out)
        del result_frames

        self._cleanup_gpu()
        return results, video_out, {
            "total": total,
            "elapsed": elapsed,
            "avg_time": elapsed / total if total else 0,
            "output_dir": out_dir,
            "output_video": video_out,
        }

    # ==================================================================
    # 视频 — 文本跟踪
    # ==================================================================

    def track_video_text(
        self,
        video_path: str,
        text: str,
        mask_mode: Optional[bool] = None,
        output_path: Optional[str] = None,
        callback=None,
    ) -> Tuple[str, dict]:
        """使用文本提示跟踪视频

        Args:
            video_path: 视频路径
            text: 文本提示
            mask_mode: 二值 mask
            output_path: 输出视频路径 (默认自动生成)
            callback: 进度回调 callback(stage, current, total)
                      stage: "detect" / "propagate" / "render"

        Returns:
            (output_video_path, info_dict)
        """
        self._ensure_mode("video")
        mmode = mask_mode if mask_mode is not None else self.mask_mode

        predictor = self._get_video_predictor()
        frames, fps = _read_video_frames(video_path)
        total = len(frames)
        h, w = frames[0].shape[:2]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            resp = predictor.handle_request({
                "type": "start_session", "resource_path": video_path,
            })
        session_id = resp["session_id"]

        try:
            if callback:
                callback("detect", 0, total)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                response = predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": text.strip(),
                })
            first_output = response.get("outputs", {})
            obj_ids = first_output.get("out_obj_ids", [])
            n_objects = len(obj_ids) if hasattr(obj_ids, "__len__") else 0
            if n_objects == 0:
                predictor.handle_request({"type": "close_session", "session_id": session_id})
                return None, {"n_objects": 0, "error": f"未检测到 \"{text}\""}

            out_path = output_path or os.path.join(
                self.output_dir, f"track_{int(time.time())}.mp4")
            result, _ = self._propagate_and_render_highlevel(
                predictor, session_id, frames, fps, first_output,
                prompt_frame_idx=0, mask_mode=mmode, output_path=out_path,
                callback=callback,
            )
        finally:
            predictor.handle_request({"type": "close_session", "session_id": session_id})

        del frames
        self._cleanup_gpu()
        return result, {"n_objects": n_objects, "total_frames": total}

    # ==================================================================
    # 视频 — 点击跟踪
    # ==================================================================

    def track_video_points(
        self,
        video_path: str,
        points: List[Tuple[int, int, int]],
        frame_idx: int = 0,
        mask_mode: Optional[bool] = None,
        output_path: Optional[str] = None,
        callback=None,
    ) -> Tuple[str, dict]:
        """使用点击提示跟踪视频

        Args:
            video_path: 视频路径
            points: [(x, y, label), ...] 像素坐标, label=1 前景 / 0 背景
            frame_idx: 标注帧索引
            mask_mode: 二值 mask
            output_path: 输出路径
            callback: 进度回调 callback(stage, current, total)

        Returns:
            (output_video_path, info_dict)
        """
        self._ensure_mode("video")
        mmode = mask_mode if mask_mode is not None else self.mask_mode

        tracker = self._get_tracker()
        frames, fps = _read_video_frames(video_path)
        total = len(frames)
        h, w = frames[0].shape[:2]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = tracker.init_state(video_path=video_path)

        pts = torch.tensor([[x / w, y / h] for x, y, _ in points], dtype=torch.float32)
        lbls = torch.tensor([l for _, _, l in points], dtype=torch.int32)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, out_obj_ids, _, _ = tracker.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=1,
                points=pts,
                labels=lbls,
                clear_old_points=False,
            )

        out_path = output_path or os.path.join(
            self.output_dir, f"track_{int(time.time())}.mp4")
        result = self._tracker_propagate_and_render(
            tracker, inference_state, frames, fps,
            mask_mode=mmode, output_path=out_path, callback=callback,
        )

        del inference_state, frames
        self._cleanup_gpu()
        return result, {"total_frames": total}

    # ==================================================================
    # 视频 — 框选跟踪
    # ==================================================================

    def track_video_box(
        self,
        video_path: str,
        box: Optional[Tuple[int, int, int, int]] = None,
        text: str = "",
        frame_idx: int = 0,
        mask_mode: Optional[bool] = None,
        output_path: Optional[str] = None,
        callback=None,
        boxes: Optional[List[tuple]] = None,
    ) -> Tuple[str, dict]:
        """使用框选提示跟踪视频，可选结合文本，支持多框 + 正/负向框。

        Args:
            video_path: 视频路径
            box: (x1,y1,x2,y2) 单框；与 ``boxes`` 互斥（兼容旧 API）
            text: 可选文本提示
            frame_idx: 标注帧索引
            mask_mode: 二值 mask
            output_path: 输出路径
            callback: 进度回调
            boxes: 多框列表，每项 (x1,y1,x2,y2) 或 (x1,y1,x2,y2,is_pos)；
                   至少需要 1 个正向框

        路径选择：
            SAM3.1 / 有文本 / 多框 / 含负向框 → 高层 handle_request API（分步
                add_prompt：第 1 框作初始 visual prompt，其余对同一 obj_id 做
                refinement, clear_old_boxes=False）。
            SAM3 + 无文本 + 单正向框 → 底层 tracker.add_new_points_or_box。

        Returns:
            (output_video_path, info_dict)
        """
        self._ensure_mode("video")
        mmode = mask_mode if mask_mode is not None else self.mask_mode
        text = (text or "").strip()

        # ---- 归一化框输入 ----
        raw_boxes: List[tuple] = []
        if boxes:
            raw_boxes.extend(boxes)
        if box is not None:
            raw_boxes.append(box)
        if not raw_boxes:
            return None, {"error": "至少需要 1 个框"}
        norm_boxes: List[Tuple[int, int, int, int, bool]] = []
        for b in raw_boxes:
            if len(b) >= 5:
                x1, y1, x2, y2, is_pos = int(b[0]), int(b[1]), int(b[2]), int(b[3]), bool(b[4])
            else:
                x1, y1, x2, y2 = (int(b[i]) for i in range(4))
                is_pos = True
            norm_boxes.append((x1, y1, x2, y2, is_pos))
        pos_boxes = [b for b in norm_boxes if b[4]]
        neg_boxes = [b for b in norm_boxes if not b[4]]
        if not pos_boxes:
            return None, {"error": "至少需要 1 个正向框"}

        out_path = output_path or os.path.join(
            self.output_dir, f"track_{int(time.time())}.mp4")

        use_high_level = (
            bool(text) or self.version == "sam3.1"
            or len(norm_boxes) > 1 or len(neg_boxes) > 0
        )

        if use_high_level:
            # ---- 高层 handle_request API ----
            predictor = self._get_video_predictor()
            frames, fps = _read_video_frames(video_path)
            total = len(frames)
            h, w = frames[0].shape[:2]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                resp = predictor.handle_request({
                    "type": "start_session", "resource_path": video_path,
                })
            session_id = resp["session_id"]

            # 归一化坐标 → xywh 0~1
            bounding_boxes = []
            bounding_box_labels = []
            for x1, y1, x2, y2, is_pos in norm_boxes:
                bounding_boxes.append([x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h])
                bounding_box_labels.append(1 if is_pos else 0)

            try:
                # SAM3 视频 predictor 的初始 visual prompt 一次只允许 1 个 box
                # （sam3_video_inference.py::_get_visual_prompt），无论是否有文本。
                # 多框时分步：第 1 框初始 + 其余对同一 obj_id 做 refinement
                # （必须 obj_id=refine_obj_id + clear_old_boxes=False）。
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    first_req = {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": frame_idx,
                        "bounding_boxes": [bounding_boxes[0]],
                        "bounding_box_labels": [bounding_box_labels[0]],
                    }
                    if text:
                        first_req["text"] = text
                    response = predictor.handle_request(first_req)

                    first_outputs = response.get("outputs") or {}
                    first_obj_ids = first_outputs.get("out_obj_ids", [])
                    if hasattr(first_obj_ids, "tolist"):
                        first_obj_ids = first_obj_ids.tolist()
                    refine_obj_id = int(first_obj_ids[0]) if len(first_obj_ids) > 0 else 1

                    for i in range(1, len(bounding_boxes)):
                        response = predictor.handle_request({
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": frame_idx,
                            "bounding_boxes": [bounding_boxes[i]],
                            "bounding_box_labels": [bounding_box_labels[i]],
                            "obj_id": refine_obj_id,
                            "clear_old_boxes": False,
                        })

                first_output = response.get("outputs", {})
                obj_ids = first_output.get("out_obj_ids", [])
                n_objects = len(obj_ids) if hasattr(obj_ids, "__len__") else 0
                if n_objects == 0:
                    predictor.handle_request({"type": "close_session", "session_id": session_id})
                    return None, {"n_objects": 0, "error": "未检测到目标"}

                result, _ = self._propagate_and_render_highlevel(
                    predictor, session_id, frames, fps, first_output,
                    prompt_frame_idx=frame_idx, mask_mode=mmode,
                    output_path=out_path, callback=callback,
                )
            finally:
                predictor.handle_request({"type": "close_session", "session_id": session_id})

            del frames
            self._cleanup_gpu()
            return result, {
                "n_objects": n_objects, "total_frames": total,
                "text": text, "n_pos": len(pos_boxes), "n_neg": len(neg_boxes),
            }

        else:
            # ---- SAM3 + 无文本 + 单正向框：底层 tracker API ----
            x1, y1, x2, y2 = pos_boxes[0][:4]
            tracker = self._get_tracker()
            frames, fps = _read_video_frames(video_path)
            total = len(frames)
            h, w = frames[0].shape[:2]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                inference_state = tracker.init_state(video_path=video_path)

            box_tensor = torch.tensor(
                [[x1 / w, y1 / h, x2 / w, y2 / h]], dtype=torch.float32,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, out_obj_ids, _, _ = tracker.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=1,
                    box=box_tensor,
                )

            result = self._tracker_propagate_and_render(
                tracker, inference_state, frames, fps,
                mask_mode=mmode, output_path=out_path, callback=callback,
            )

            del inference_state, frames
            self._cleanup_gpu()
            return result, {"total_frames": total}

    # ==================================================================
    # 内部：高层 API 传播 + 渲染
    # ==================================================================

    def _propagate_and_render_highlevel(
        self, predictor, session_id, frames, fps, first_output,
        prompt_frame_idx=0, mask_mode=False, output_path=None,
        callback=None,
    ) -> Tuple[str, None]:
        total = len(frames)
        h, w = frames[0].shape[:2]
        outputs_per_frame = {prompt_frame_idx: first_output}

        with torch.autocast("cuda", dtype=torch.bfloat16):
            for resp in predictor.handle_stream_request({
                "type": "propagate_in_video",
                "session_id": session_id,
            }):
                fidx = resp["frame_index"]
                outputs_per_frame[fidx] = resp["outputs"]
                if callback:
                    callback("propagate", len(outputs_per_frame), total)

        rendered = []
        for i in range(total):
            frame = frames[i]
            if i in outputs_per_frame:
                r = video_masks_to_binary(outputs_per_frame[i], h, w) if mask_mode else overlay_video_masks(frame, outputs_per_frame[i])
            else:
                r = np.zeros((h, w, 3), dtype=np.uint8) if mask_mode else frame
            rendered.append(r)
            if callback and i % max(total // 20, 1) == 0:
                callback("render", i, total)

        del outputs_per_frame
        _write_video(rendered, fps, output_path)
        del rendered
        return output_path, None

    # ==================================================================
    # 内部：底层 tracker 传播 + 渲染
    # ==================================================================

    def _tracker_propagate_and_render(
        self, tracker, inference_state, frames, fps,
        mask_mode=False, output_path=None, callback=None,
    ) -> str:
        total = len(frames)
        h, w = frames[0].shape[:2]
        masks_per_frame = {}

        with torch.autocast("cuda", dtype=torch.bfloat16):
            for fidx, obj_ids, low_res, video_res, scores in tracker.propagate_in_video(
                inference_state,
                start_frame_idx=0,
                max_frame_num_to_track=total,
                reverse=False,
                propagate_preflight=True,
            ):
                masks_per_frame[fidx] = (obj_ids, video_res)
                if callback:
                    callback("propagate", len(masks_per_frame), total)

        rendered = []
        for i in range(total):
            frame = frames[i]
            if i in masks_per_frame:
                obj_ids, video_res = masks_per_frame[i]
                if mask_mode:
                    combined = np.zeros((h, w), dtype=np.uint8)
                    for j in range(video_res.shape[0]):
                        m = video_res[j].cpu().numpy().squeeze()
                        if m.shape != (h, w):
                            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                        combined[m > 0.0] = 255
                    r = np.stack([combined] * 3, axis=-1)
                else:
                    binary_masks = (video_res > 0.0).cpu()
                    outputs = {
                        "out_obj_ids": obj_ids if not isinstance(obj_ids, list) else obj_ids,
                        "out_binary_masks": binary_masks,
                    }
                    r = overlay_video_masks(frame, outputs)
            else:
                r = np.zeros((h, w, 3), dtype=np.uint8) if mask_mode else frame
            rendered.append(r)
            if callback and i % max(total // 20, 1) == 0:
                callback("render", i, total)

        del masks_per_frame
        _write_video(rendered, fps, output_path)
        del rendered
        return output_path

    # ------ 工具 ------

    @staticmethod
    def _to_pil(image) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise TypeError(f"不支持的图像类型: {type(image)}")

    def extract_frame(self, video_path: str, frame_idx: int = 0) -> np.ndarray:
        """从视频提取指定帧 (RGB numpy)"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"无法读取第 {frame_idx} 帧")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_video_info(self, video_path: str) -> dict:
        """获取视频基本信息"""
        cap = cv2.VideoCapture(video_path)
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        cap.release()
        return info


# ===================================================================
# CLI 入口
# ===================================================================

def _progress_printer(stage, current, total):
    """简单的 CLI 进度回调"""
    bar_len = 30
    pct = current / max(total, 1)
    filled = int(bar_len * pct)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r  [{bar}] {stage}: {current}/{total}", end="", flush=True)
    if current >= total:
        print()


def _batch_progress(current, total, name):
    bar_len = 30
    pct = (current + 1) / max(total, 1)
    filled = int(bar_len * pct)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r  [{bar}] {current+1}/{total} {name}", end="", flush=True)
    if current + 1 >= total:
        print()


def _parse_points(raw: list) -> List[Tuple[int, int, int]]:
    """解析 "x,y,label" 字符串列表 → [(x, y, label), ...]"""
    pts = []
    for s in raw:
        parts = s.split(",")
        if len(parts) != 3:
            raise ValueError(f"点格式错误: '{s}', 应为 x,y,label (如 200,150,1)")
        pts.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return pts


def _parse_box(raw: str) -> Tuple[int, ...]:
    """解析 "x1,y1,x2,y2" 或 "x1,y1,x2,y2,is_pos"。

    is_pos: 1=正向框 (默认)，0=负向框。
    """
    parts = raw.split(",")
    if len(parts) == 4:
        return tuple(int(p) for p in parts)
    if len(parts) == 5:
        x1, y1, x2, y2 = (int(parts[i]) for i in range(4))
        return (x1, y1, x2, y2, bool(int(parts[4])))
    raise ValueError(f"框格式错误: '{raw}', 应为 x1,y1,x2,y2 或 x1,y1,x2,y2,is_pos")


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 推理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python inference.py image-text -i photo.jpg -t "person, car"
  python inference.py image-box -i photo.jpg --box 100,50,400,300
  python inference.py image-points -i photo.jpg --points 200,150,1 350,200,0
  python inference.py batch -d ./images -t "person"
  python inference.py batch -v input.mp4 -t "car" --interval 5
  python inference.py video-text -v input.mp4 -t "person"
  python inference.py video-points -v input.mp4 --points 200,150,1 --frame 30
  python inference.py video-box -v input.mp4 --box 100,50,400,300 -t "person"
""",
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    # 通用参数
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="sam3", choices=["sam3", "sam3.1"], help="模型版本")
    common.add_argument("--mask", action="store_true", help="输出二值 mask 而非叠加可视化")
    common.add_argument("--no-fa", action="store_true", help="禁用 Flash Attention")
    common.add_argument("--mmgp", action="store_true", help="启用 mmgp 显存卸载")
    common.add_argument("--mmgp-profile", type=int, default=4, help="mmgp profile（默认 4）")
    common.add_argument("--sam31-batch-size", type=int, default=4, help="SAM3.1 backbone 批大小（官方默认 16，本工具默认 4 兼顾显存；mmgp 建议 1）")
    common.add_argument("-o", "--output", help="输出路径")

    # image-text
    p = sub.add_parser("image-text", parents=[common], help="图像文本分割")
    p.add_argument("-i", "--image", required=True, help="输入图片")
    p.add_argument("-t", "--text", required=True, help="文本提示")
    p.add_argument("--confidence", type=float, default=0.5, help="置信度阈值")

    # image-box
    p = sub.add_parser("image-box", parents=[common], help="图像框选分割")
    p.add_argument("-i", "--image", required=True, help="输入图片")
    p.add_argument("--box", required=True, action="append", help="框 x1,y1,x2,y2 (可多个)")
    p.add_argument("-t", "--text", default="", help="可选文本提示")
    p.add_argument("--confidence", type=float, default=0.5, help="置信度阈值")

    # image-points
    p = sub.add_parser("image-points", parents=[common], help="图像点击分割")
    p.add_argument("-i", "--image", required=True, help="输入图片")
    p.add_argument("--points", nargs="+", required=True, help="点 x,y,label (1=前景 0=背景)")

    # batch
    p = sub.add_parser("batch", parents=[common], help="批量分割")
    p.add_argument("-d", "--dir", help="图片文件夹")
    p.add_argument("-v", "--video", help="视频文件")
    p.add_argument("-t", "--text", required=True, help="文本提示")
    p.add_argument("--confidence", type=float, default=0.5, help="置信度阈值")
    p.add_argument("--interval", type=int, default=1, help="视频抽帧间隔")

    # video-text
    p = sub.add_parser("video-text", parents=[common], help="视频文本跟踪")
    p.add_argument("-v", "--video", required=True, help="输入视频")
    p.add_argument("-t", "--text", required=True, help="文本提示")

    # video-points
    p = sub.add_parser("video-points", parents=[common], help="视频点击跟踪")
    p.add_argument("-v", "--video", required=True, help="输入视频")
    p.add_argument("--points", nargs="+", required=True, help="点 x,y,label")
    p.add_argument("--frame", type=int, default=0, help="标注帧索引")

    # video-box
    p = sub.add_parser("video-box", parents=[common], help="视频框选跟踪")
    p.add_argument("-v", "--video", required=True, help="输入视频")
    p.add_argument("--box", action="append", default=[],
                   help="正向框 x1,y1,x2,y2 (可多个，至少 1 个)")
    p.add_argument("--neg-box", action="append", default=[],
                   help="负向框 x1,y1,x2,y2 (可多个)")
    p.add_argument("-t", "--text", default="", help="可选文本提示")
    p.add_argument("--frame", type=int, default=0, help="标注帧索引")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sam = SAM3Inference(
        version=args.model,
        use_fa=not args.no_fa,
        mask_mode=args.mask,
        use_mmgp=args.mmgp,
        mmgp_profile=args.mmgp_profile,
        sam31_batch_size=args.sam31_batch_size,
    )

    try:
        if args.command == "image-text":
            sam.confidence = args.confidence
            out_path = args.output or os.path.join(OUTPUT_DIR, f"seg_{int(time.time())}.png")
            result, info = sam.segment_image_text(args.image, args.text)
            Image.fromarray(result).save(out_path)
            print(f"检测到 {info['n_objects']} 个对象, 耗时 {info['elapsed']:.2f}s")
            print(f"已保存: {out_path}")

        elif args.command == "image-box":
            sam.confidence = args.confidence
            boxes = [_parse_box(b) for b in args.box]
            out_path = args.output or os.path.join(OUTPUT_DIR, f"seg_{int(time.time())}.png")
            result, info = sam.segment_image_box(args.image, boxes, text=args.text)
            Image.fromarray(result).save(out_path)
            print(f"检测到 {info['n_objects']} 个对象, 耗时 {info['elapsed']:.2f}s")
            print(f"已保存: {out_path}")

        elif args.command == "image-points":
            points = _parse_points(args.points)
            out_path = args.output or os.path.join(OUTPUT_DIR, f"seg_{int(time.time())}.png")
            result, info = sam.segment_image_points(args.image, points)
            Image.fromarray(result).save(out_path)
            print(f"最佳分数: {info['best_score']:.3f}, 耗时 {info['elapsed']:.2f}s")
            print(f"已保存: {out_path}")

        elif args.command == "batch":
            sam.confidence = args.confidence
            if args.dir:
                results, info = sam.batch_segment_folder(
                    args.dir, args.text,
                    output_dir=args.output,
                    callback=_batch_progress,
                )
                print(f"\n{info['total']} 张, 耗时 {info['elapsed']:.1f}s, "
                      f"平均 {info['avg_time']:.2f}s/张")
                print(f"结果: {info['output_dir']}")
            elif args.video:
                results, video_out, info = sam.batch_segment_video(
                    args.video, args.text,
                    interval=args.interval,
                    output_dir=args.output,
                    callback=_batch_progress,
                )
                print(f"\n{info['total']} 帧, 耗时 {info['elapsed']:.1f}s")
                print(f"结果: {info['output_dir']}")
                if video_out:
                    print(f"视频: {video_out}")
            else:
                print("错误: 需要 -d (文件夹) 或 -v (视频)")
                return

        elif args.command == "video-text":
            out_path = args.output or os.path.join(OUTPUT_DIR, f"track_{int(time.time())}.mp4")
            result, info = sam.track_video_text(
                args.video, args.text,
                output_path=out_path, callback=_progress_printer,
            )
            if result:
                print(f"跟踪完成: {info['n_objects']} 个对象, {info['total_frames']} 帧")
                print(f"已保存: {result}")
            else:
                print(f"跟踪失败: {info.get('error', '未知')}")

        elif args.command == "video-points":
            points = _parse_points(args.points)
            out_path = args.output or os.path.join(OUTPUT_DIR, f"track_{int(time.time())}.mp4")
            result, info = sam.track_video_points(
                args.video, points,
                frame_idx=args.frame,
                output_path=out_path, callback=_progress_printer,
            )
            print(f"跟踪完成: {info['total_frames']} 帧")
            print(f"已保存: {result}")

        elif args.command == "video-box":
            pos_boxes = [_parse_box(b) for b in args.box]
            neg_boxes = []
            for b in args.neg_box:
                parsed = _parse_box(b)
                # 强制为负向框
                neg_boxes.append((parsed[0], parsed[1], parsed[2], parsed[3], False))
            # 保证 pos 是 5 元组，is_pos=True
            norm_pos = []
            for b in pos_boxes:
                if len(b) == 5:
                    norm_pos.append((b[0], b[1], b[2], b[3], True if bool(b[4]) else False))
                else:
                    norm_pos.append((b[0], b[1], b[2], b[3], True))
            all_boxes = norm_pos + neg_boxes
            if not all_boxes:
                print("错误: --box 或 --neg-box 至少提供 1 个")
                return
            out_path = args.output or os.path.join(OUTPUT_DIR, f"track_{int(time.time())}.mp4")
            result, info = sam.track_video_box(
                args.video, boxes=all_boxes,
                text=args.text,
                frame_idx=args.frame,
                output_path=out_path, callback=_progress_printer,
            )
            if result:
                desc_parts = [f"{sum(1 for b in all_boxes if b[4])}正"]
                n_neg = sum(1 for b in all_boxes if not b[4])
                if n_neg:
                    desc_parts.append(f"{n_neg}负")
                print(f"跟踪完成 ({'+'.join(desc_parts)}): {info.get('total_frames', '?')} 帧")
                print(f"已保存: {result}")
            else:
                print(f"跟踪失败: {info.get('error', '未知')}")

    finally:
        sam.unload_all()


if __name__ == "__main__":
    main()
