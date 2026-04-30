"""
SAM3 Gradio 交互式分割工具
功能：
  1. 图像文本分割 — 输入文本描述，自动分割图像中匹配的对象
  2. 图像框选分割 — 在图片上画框圈选目标进行分割（支持正/负框）
  3. 视频目标跟踪 — 支持文本/点击/框选提示，支持 SAM3 和 SAM3.1 模型
  4. 交互式点击分割 — 在图像上点击标记前景/背景点进行精确分割
  5. 批量图像分割 — 输入图片文件夹或视频，批量执行文本提示分割
"""

import gc
import os
import subprocess
import sys
import time
import traceback

import cv2
import gradio as gr
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

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "sam3"))
SAM3_DIR = os.path.join(BASE_DIR, "sam3")
CHECKPOINT_SAM3 = os.path.join(SAM3_DIR, "checkpoints", "sam3.pt")
CHECKPOINT_SAM31 = os.path.join(SAM3_DIR, "checkpoints", "sam3.1_multiplex.pt")
BPE_PATH = os.path.join(SAM3_DIR, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 启动诊断
# ============================================================
print(f"[SAM3] 工作目录: {BASE_DIR}")
print(f"[SAM3] Python:    {sys.executable}")
print(f"[SAM3] 设备:      {DEVICE}")
print(f"[SAM3] SAM3 ckpt:   {CHECKPOINT_SAM3}  (存在: {os.path.isfile(CHECKPOINT_SAM3)})")
print(f"[SAM3] SAM3.1 ckpt: {CHECKPOINT_SAM31}  (存在: {os.path.isfile(CHECKPOINT_SAM31)})")
print(f"[SAM3] BPE:          {BPE_PATH}  (存在: {os.path.isfile(BPE_PATH)})")

# ============================================================
# 全局模型（懒加载）
# ============================================================
_image_processor = None
_interactive_model = None
_interactive_processor = None
_video_predictors = {}  # key: "sam3" or "sam3.1"
_video_use_fa = True   # 当前视频模型是否使用 Flash Attention
_active_mode = None  # "image_sam3", "image_sam3.1", "interactive_sam3", "interactive_sam3.1", "video_sam3", "video_sam3.1"
_mmgp_enabled = False
_mmgp_profile = 4
_mmgp_applied = {}  # target_name -> offload_obj (用于后续 release pinned host memory)

COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
    (60, 100, 170), (200, 80, 120), (100, 200, 100), (180, 120, 60),
]


def _cleanup_gpu():
    """释放 GPU 缓存与 pinned host memory（共享 GPU 显存）。

    注意：torch.cuda.empty_cache() 只清 CUDA 设备缓存，不清 pinned host
    memory；后者要通过 torch._C._host_emptyCache() 才能归还给 OS。
    在 Windows 上还需要 EmptyWorkingSet 把进程工作集压回去，任务管理器
    的「共享 GPU 内存」才会下降。优先用 mmgp 提供的 flush_torch_caches。
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

    # mmgp 不可用时的回退：手动清 pinned host cache + Windows 工作集
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


def _release_mmgp_for(prefix):
    """Release mmgp offload objects whose target_name starts with prefix.
    This frees pinned host memory (counted as Windows '共享 GPU' usage)."""
    if not _mmgp_applied:
        return
    keys = [k for k in list(_mmgp_applied.keys()) if k.startswith(prefix)]
    for k in keys:
        obj = _mmgp_applied.pop(k, None)
        if obj is None:
            continue
        try:
            release_fn = getattr(obj, "release", None)
            if callable(release_fn):
                release_fn()
                print(f"[MMGP] 已释放 offload ({k})")
        except Exception as e:
            print(f"[MMGP] 释放 offload 失败 ({k}): {e}")


def _unload_model(name):
    """卸载指定模型并释放显存（含 mmgp pinned host memory）"""
    global _image_processor, _interactive_model, _interactive_processor, _video_predictors
    if name.startswith("image") and _image_processor is not None:
        print("[SAM3] 卸载图像分割模型...")
        del _image_processor
        _image_processor = None
        _release_mmgp_for("image.")
    elif name.startswith("interactive") and _interactive_model is not None:
        print("[SAM3] 卸载交互式分割模型...")
        del _interactive_model, _interactive_processor
        _interactive_model = None
        _interactive_processor = None
        _release_mmgp_for("interactive.")
    elif name.startswith("video_"):
        ver = name.replace("video_", "")  # "sam3" or "sam3.1"
        if ver in _video_predictors:
            print(f"[SAM3] 卸载视频模型 ({ver})...")
            del _video_predictors[ver]
            _release_mmgp_for(f"video.{ver}.")


def _ensure_mode(mode):
    """确保当前模式的模型已加载，并卸载其他模式的模型以释放显存。
    mode: "image_sam3", "image_sam3.1", "interactive_sam3", "interactive_sam3.1", "video_sam3", "video_sam3.1"
    注意：image_sam3 / image_sam3.1 共用同一个 _image_processor；
          interactive_sam3 / interactive_sam3.1 共用同一个 _interactive_model。
          同类变量不应互相卸载，否则会把自己删除。
    """
    global _active_mode
    if _active_mode == mode:
        return

    all_modes = {"image_sam3", "image_sam3.1", "interactive_sam3", "interactive_sam3.1", "video_sam3", "video_sam3.1"}
    # 排除与目标共用同一变量的兄弟模式，避免把自己卸载
    # 注意：image_* 与 interactive_* 通过「加载图像模型」按钮一起加载，应共存而不互相卸载，
    # 否则用了点击分割后再做文本/框选/批量分割时 _image_processor 会变成 None。
    if mode.startswith("interactive") or mode.startswith("image"):
        same_var = {"image_sam3", "image_sam3.1", "interactive_sam3", "interactive_sam3.1"}
    else:
        same_var = set()
    for m in all_modes - {mode} - same_var:
        _unload_model(m)

    _cleanup_gpu()
    _active_mode = mode
    vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"[SAM3] 模式切换 → {mode} (当前显存: {vram:.0f} MB)")


def _set_mmgp_config(enabled=False, profile=4):
    global _mmgp_enabled, _mmgp_profile
    _mmgp_enabled = bool(enabled)
    _mmgp_profile = int(profile)
    if _mmgp_enabled and not _MMGP_AVAILABLE:
        print("[MMGP] 警告: 未检测到 mmgp，开关将被忽略")


def _get_mmgp_profile():
    if not _MMGP_AVAILABLE:
        return None

    profile_names = {
        1: "HighRAM_HighVRAM",
        2: "HighRAM_LowVRAM",
        3: "LowRAM_HighVRAM",
        4: "LowRAM_LowVRAM",
    }
    preferred = [
        profile_names.get(int(_mmgp_profile)),
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


def _apply_mmgp_to(module, target_name, module_key="model", **override_kwargs):
    if not _mmgp_enabled:
        return
    if not _MMGP_AVAILABLE:
        return
    if module is None:
        return

    key = target_name
    if key in _mmgp_applied:
        return

    profile = _get_mmgp_profile()
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

    _mmgp_applied[key] = offload_obj
    try:
        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")
    except Exception:
        pass
    if override_kwargs:
        print(f"[MMGP] 已应用到 {target_name} (profile={_mmgp_profile}, overrides={override_kwargs})")
    else:
        print(f"[MMGP] 已应用到 {target_name} (profile={_mmgp_profile})")


def _apply_mmgp_to_video_predictor(predictor, version, sam31_batch_size=1):
    if not _mmgp_enabled:
        return
    model = getattr(predictor, "model", None)
    if model is None:
        return

    def _param_mb(mod):
        try:
            return sum(p.numel() * p.element_size() for p in mod.parameters()) / 1024**2
        except Exception:
            return 0.0

    # 在 mmgp 卸载参数前，强制触发 _device 缓存为 CUDA
    for _obj in [model, getattr(model, "detector", None)]:
        if _obj is not None:
            try:
                if isinstance(getattr(type(_obj), "device", None), property):
                    _cached = _obj.device
                    if _cached is not None and str(_cached) != "cpu":
                        _obj._device = _cached
            except Exception:
                pass

    # 1. 挂载 detector.backbone（vision_backbone + language_backbone）
    _detector = getattr(model, "detector", None)
    _bb = getattr(_detector, "backbone", None) if _detector is not None else None
    if _bb is not None:
        _bb_mb = _param_mb(_bb)
        print(f"[MMGP] 准备卸载 detector.backbone ({_bb_mb:.0f} MB) 到 RAM ...")
        _apply_mmgp_to(
            _bb,
            f"video.{version}.detector.backbone",
            module_key="transformer",
            quantizeTransformer=False,
            asyncTransfers=False,
        )

    # 2. 挂载 tracker
    tracker = getattr(model, "tracker", None)
    if tracker is not None:
        _trk_mb = _param_mb(tracker)
        print(f"[MMGP] 准备卸载 tracker ({_trk_mb:.0f} MB) 到 RAM ...")
        _apply_mmgp_to(
            tracker,
            f"video.{version}.tracker",
            module_key="transformer",
            quantizeTransformer=False,
            asyncTransfers=False,
        )

    # 3. SAM3.1 专项：把 tracker 推理状态按帧卸载到 CPU RAM
    #    原因：SAM3.1 tracker 默认把每帧的 maskmem_features、pred_masks、
    #    image_features 全部留在 GPU（offload_output_to_cpu_for_eval=False）。
    #    随着视频帧数增加，这些中间张量会持续积累，导致显存拉满。
    #    代码已预留 .cpu()/.cuda() 的完整 offload 流程，只需打开此开关即可。
    if version == "sam3.1":
        # tracker = Sam3MultiplexPredictorWrapper → .model = Sam3VideoTrackingMultiplexDemo
        _inner = getattr(tracker, "model", None) if tracker is not None else None
        if _inner is not None and hasattr(_inner, "offload_output_to_cpu_for_eval"):
            _inner.offload_output_to_cpu_for_eval = True
            print("[MMGP] SAM3.1 tracker: 已启用 offload_output_to_cpu_for_eval=True（推理状态按帧卸载到 CPU RAM）")
        # 同步设置 fill_hole_area=0 下的 non-cond 帧裁剪，进一步限制显存积累
        if _inner is not None and hasattr(_inner, "trim_past_non_cond_mem_for_eval"):
            _inner.trim_past_non_cond_mem_for_eval = True
            print("[MMGP] SAM3.1 tracker: 已启用 trim_past_non_cond_mem_for_eval=True（裁剪旧非条件帧记忆）")

    # 4. SAM3.1 专项：控制批量 grounding 大小以平衡显存与速度
    #    batched_grounding_batch_size=16（默认）: backbone 一次处理 16 帧，速度快但激活内存 ×16（~1.5 GB）
    #    batched_grounding_batch_size=1（mmgp 建议）: 逐帧处理，显存降至与 SAM3 相当
    if version == "sam3.1" and model is not None:
        _eff_batch = int(sam31_batch_size)
        _use_batched = _eff_batch > 1
        if hasattr(model, "use_batched_grounding"):
            model.use_batched_grounding = _use_batched
            print(f"[MMGP] SAM3.1: use_batched_grounding={_use_batched}（batch={_eff_batch}）")
        if hasattr(model, "batched_grounding_batch_size"):
            model.batched_grounding_batch_size = _eff_batch
        if hasattr(model, "postprocess_batch_size"):
            model.postprocess_batch_size = _eff_batch
            print(f"[MMGP] SAM3.1: postprocess_batch_size={_eff_batch}")


def get_image_processor(version="sam3"):
    global _image_processor
    if _image_processor is None:
        try:
            ckpt = CHECKPOINT_SAM31 if version == "sam3.1" else CHECKPOINT_SAM3
            print(f"[SAM3] 正在导入图像模型 ({version})...")
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            print(f"[SAM3] 导入成功，正在加载图像分割模型 ({version})...")
            # 始终先在 GPU 加载：与视频模型一致，便于在「共享 GPU」中观察 mmgp 卸载行为。
            # mmgp 之后会把权重搬回 RAM；CUDA caching allocator 保留的闲置显存块会被驱动
            # 计入「共享 GPU 内存」（专用显存仍会下降），与视频模型表现一致。
            model = build_sam3_image_model(
                bpe_path=BPE_PATH,
                checkpoint_path=ckpt,
                device=DEVICE,
            )
            # pinnedMemory=True 强制全部权重使用 pinned host memory，任务管理器能看到「共享 GPU」上涨。
            # 不使用 module_key="transformer"，避免 profile 默认 budgets["transformer"]=1200 把子模块滞留 CPU
            # （predict_inst 会绕过顶层 forward 直接调用 sam_prompt_encoder，导致 mmgp hook 不触发）。
            _apply_mmgp_to(model, f"image.{version}.model", quantizeTransformer=False, asyncTransfers=False, pinnedMemory=True, budgets=None)
            if _mmgp_enabled and _MMGP_AVAILABLE:
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
            _image_processor = Sam3Processor(model, confidence_threshold=0.5)
            print(f"[SAM3] 图像分割模型 ({version}) 加载完成 ✓")
        except Exception as e:
            print(f"[SAM3] ✗ 图像模型加载失败: {e}")
            traceback.print_exc()
            raise
    return _image_processor


def get_video_predictor(version="sam3", use_fa3=True):
    global _video_predictors, _video_use_fa
    # 如果 FA 设置变化，卸载已缓存的视频模型并重建
    if use_fa3 != _video_use_fa:
        for ver in list(_video_predictors.keys()):
            print(f"[SAM3] Flash Attention 设置变更，卸载视频模型 ({ver})...")
            del _video_predictors[ver]
        _video_predictors.clear()
        _video_use_fa = use_fa3
        _cleanup_gpu()
    if version not in _video_predictors:
        try:
            fa_label = "FA2" if use_fa3 else "SDPA"
            print(f"[SAM3] 正在加载视频模型 ({version}, {fa_label})...")
            from sam3.model_builder import build_sam3_predictor
            ckpt = CHECKPOINT_SAM31 if version == "sam3.1" else CHECKPOINT_SAM3
            _video_predictors[version] = build_sam3_predictor(
                version=version,
                checkpoint_path=ckpt,
                bpe_path=BPE_PATH,
                use_fa3=use_fa3,
            )
            _apply_mmgp_to_video_predictor(_video_predictors[version], version)
            print(f"[SAM3] 视频跟踪模型 ({version}, {fa_label}) 加载完成 ✓")
        except Exception as e:
            print(f"[SAM3] ✗ 视频模型 ({version}) 加载失败: {e}")
            traceback.print_exc()
            raise
    return _video_predictors[version]


def get_interactive_model(version="sam3"):
    """交互式点击分割模型始终使用 sam3.pt（sam3.1_multiplex.pt 不含 inst_interactive_predictor 权重）"""
    global _interactive_model, _interactive_processor
    if _interactive_model is None:
        try:
            # 交互式模型只支持 sam3.pt，与 UI 选择的版本无关
            print(f"[SAM3] 正在加载交互式分割模型 (sam3, enable_inst_interactivity=True)...")
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            _interactive_model = build_sam3_image_model(
                bpe_path=BPE_PATH,
                checkpoint_path=CHECKPOINT_SAM3,
                enable_inst_interactivity=True,
            )
            _apply_mmgp_to(_interactive_model, "interactive.sam3.model", quantizeTransformer=False, asyncTransfers=False, pinnedMemory=True, budgets=None)
            _interactive_processor = Sam3Processor(_interactive_model)
            print(f"[SAM3] 交互式分割模型 (sam3) 加载完成 ✓")
        except Exception as e:
            print(f"[SAM3] ✗ 交互式模型加载失败: {e}")
            traceback.print_exc()
            raise
    return _interactive_model, _interactive_processor


# ============================================================
# 可视化工具
# ============================================================

def masks_to_binary_image(masks, h, w):
    """将多个 mask 合并为一张二值图：白色=物体, 黑色=背景"""
    combined = np.zeros((h, w), dtype=np.uint8)
    n = len(masks) if isinstance(masks, list) else (masks.shape[0] if hasattr(masks, 'shape') else 0)
    for i in range(n):
        mask = masks[i]
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask = mask.squeeze()
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h),
                              interpolation=cv2.INTER_NEAREST)
        combined[mask > 0.5] = 255
    return np.stack([combined] * 3, axis=-1)  # HxWx3


def video_masks_to_binary_frame(outputs, h, w):
    """将视频跟踪 outputs 中的 mask 合并为二值帧"""
    masks = outputs.get("out_binary_masks", [])
    n = len(masks) if isinstance(masks, list) else (masks.shape[0] if hasattr(masks, 'shape') else 0)
    combined = np.zeros((h, w), dtype=np.uint8)
    for i in range(n):
        mask = masks[i]
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask = mask.squeeze()
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h),
                              interpolation=cv2.INTER_NEAREST)
        combined[mask > 0.5] = 255
    return np.stack([combined] * 3, axis=-1)


def overlay_masks_on_image(image_np, masks, boxes, scores, alpha=0.45):
    overlay = image_np.copy()
    h, w = overlay.shape[:2]
    n = scores.shape[0] if isinstance(scores, torch.Tensor) else len(scores)

    for i in range(n):
        color = COLORS[i % len(COLORS)]

        mask = masks[i]
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask = mask.squeeze()
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h),
                              interpolation=cv2.INTER_NEAREST)
        mask_bool = mask > 0.5

        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_bool,
                np.clip(alpha * color[c] + (1 - alpha) * overlay[:, :, c],
                        0, 255).astype(np.uint8),
                overlay[:, :, c],
            )

        box = boxes[i]
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        x0, y0, x1, y1 = box.astype(int)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)

        score = scores[i].item() if isinstance(scores[i], torch.Tensor) else float(scores[i])
        label = f"#{i} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay, (x0, max(y0 - th - 8, 0)), (x0 + tw + 4, y0), color, -1)
        cv2.putText(overlay, label, (x0 + 2, max(y0 - 4, th + 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return overlay


def overlay_video_masks(frame_rgb, outputs, alpha=0.45):
    overlay = frame_rgb.copy()
    h, w = overlay.shape[:2]

    masks = outputs.get("out_binary_masks", [])
    obj_ids = outputs.get("out_obj_ids", [])
    boxes = outputs.get("out_boxes_xywh", [])
    probs = outputs.get("out_probs", [])

    n = len(obj_ids) if hasattr(obj_ids, '__len__') else (obj_ids.shape[0] if isinstance(obj_ids, (torch.Tensor, np.ndarray)) else 0)

    for i in range(n):
        oid = obj_ids[i]
        if isinstance(oid, torch.Tensor):
            oid = oid.item()
        oid = int(oid)
        color = COLORS[oid % len(COLORS)]

        mask = masks[i]
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask = mask.squeeze()
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h),
                              interpolation=cv2.INTER_NEAREST)
        mask_bool = mask > 0.5

        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_bool,
                np.clip(alpha * color[c] + (1 - alpha) * overlay[:, :, c],
                        0, 255).astype(np.uint8),
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
            label = f"id={oid}"
            if prob is not None:
                label += f" {prob:.2f}"
            (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x1, max(y1 - th_t - 6, 0)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 2, max(y1 - 3, th_t + 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


def draw_points_on_image(image_np, points):
    overlay = image_np.copy()
    for x, y, label in points:
        color = (0, 200, 0) if label == 1 else (200, 0, 0)
        cv2.circle(overlay, (int(x), int(y)), 6, color, -1)
        cv2.circle(overlay, (int(x), int(y)), 6, (255, 255, 255), 2)
        marker = "+" if label == 1 else "-"
        cv2.putText(overlay, marker, (int(x) + 10, int(y) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return overlay


def draw_boxes_on_image(image_np, boxes_data):
    """绘制框选标注。boxes_data: list of (x1, y1, x2, y2, is_positive)"""
    overlay = image_np.copy()
    for x1, y1, x2, y2, is_pos in boxes_data:
        color = (0, 200, 0) if is_pos else (200, 0, 0)
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        tag = "+" if is_pos else "-"
        cv2.putText(overlay, tag, (int(x1) + 4, int(y1) + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return overlay


# ============================================================
# Tab 1 — 图像文本分割
# ============================================================

def segment_image(image, text_prompt, confidence, mask_mode=False, model_version="sam3", use_mmgp=False, mmgp_profile=4):
    print(f"\n[SAM3] === 图像分割请求: prompt='{text_prompt}', conf={confidence}, mask={mask_mode}, ver={model_version} ===")
    if image is None:
        gr.Warning("请先上传图片")
        return None, "⚠ 请先上传图片"
    if not text_prompt or not text_prompt.strip():
        gr.Warning("请输入文本提示")
        return None, "⚠ 请输入文本提示"

    if _image_processor is None:
        return None, "⚠ 图像模型未加载，请先点击「加载图像模型」按钮"

    _set_mmgp_config(use_mmgp, mmgp_profile)
    _ensure_mode(f"image_{model_version}")
    try:
        processor = get_image_processor(model_version)
    except Exception as e:
        return None, f"模型加载失败: {e}"

    processor.confidence_threshold = confidence

    pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    print(f"[SAM3] 图像尺寸: {pil_image.size}, 提示: '{text_prompt.strip()}'")

    try:
        t0 = time.time()
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            state = processor.set_image(pil_image)
            state = processor.set_text_prompt(state=state, prompt=text_prompt.strip())
        print(f"[SAM3] 推理耗时: {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"[SAM3] ✗ 推理出错: {e}")
        traceback.print_exc()
        return None, f"推理出错: {e}"

    masks = state["masks"]
    boxes = state["boxes"]
    scores = state["scores"]
    n = scores.shape[0] if isinstance(scores, torch.Tensor) else len(scores)

    print(f"[SAM3] 检测结果: {n} 个对象")
    if n == 0:
        return image, f"未检测到与 \"{text_prompt}\" 匹配的对象（可尝试降低置信度）"

    img_np = np.array(pil_image) if isinstance(pil_image, Image.Image) else image
    if mask_mode:
        h, w = img_np.shape[:2]
        result = masks_to_binary_image(masks, h, w)
    else:
        result = overlay_masks_on_image(img_np, masks, boxes, scores)

    details = ", ".join([f"#{i}({scores[i].item():.2f})" for i in range(n)])
    print(f"[SAM3] ✓ 图像分割完成: {details}")
    _cleanup_gpu()
    return result, f"✅ 检测到 {n} 个对象: {details}"


# ============================================================
# Tab 2 — 图像框选分割 (Box Prompt)
# ============================================================

def segment_image_with_boxes(original_image, boxes_data, text_prompt, confidence, mask_mode=False, model_version="sam3", use_mmgp=False, mmgp_profile=4):
    """使用框选提示进行图像分割，可选结合文本提示"""
    text_prompt = (text_prompt or "").strip()
    print(f"\n[SAM3] === 框选分割请求: {len(boxes_data) if boxes_data else 0} 个框, text='{text_prompt}', mask={mask_mode}, ver={model_version} ===")
    if original_image is None:
        gr.Warning("请先上传图片")
        return None, "⚠ 请先上传图片"
    if not boxes_data:
        gr.Warning("请先画框标记目标区域")
        return None, "⚠ 请先画框标记目标区域"

    if _image_processor is None:
        return None, "⚠ 图像模型未加载，请先点击「加载图像模型」按钮"

    _set_mmgp_config(use_mmgp, mmgp_profile)
    _ensure_mode(f"image_{model_version}")
    try:
        processor = get_image_processor(model_version)
    except Exception as e:
        return None, f"模型加载失败: {e}"

    processor.confidence_threshold = confidence

    pil_image = Image.fromarray(original_image)
    w, h = pil_image.size

    try:
        t0 = time.time()
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            state = processor.set_image(pil_image)
            processor.reset_all_prompts(state)

            # 可选：先用文本检测候选，再用框精确指定
            if text_prompt:
                state = processor.set_text_prompt(state=state, prompt=text_prompt)

            for x1, y1, x2, y2, is_pos in boxes_data:
                # 像素 xyxy → 归一化 cxcywh
                cx = (x1 + x2) / 2.0 / w
                cy = (y1 + y2) / 2.0 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                norm_box = [cx, cy, bw, bh]
                state = processor.add_geometric_prompt(
                    state=state, box=norm_box, label=is_pos,
                )

        print(f"[SAM3] 推理耗时: {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"[SAM3] ✗ 框选分割出错: {e}")
        traceback.print_exc()
        return None, f"分割出错: {e}"

    masks = state["masks"]
    boxes = state["boxes"]
    scores = state["scores"]
    n = scores.shape[0] if isinstance(scores, torch.Tensor) else len(scores)

    print(f"[SAM3] 原始检测: {n} 个对象")

    # 过滤：只保留中心点落在用户正向框内的检测结果
    pos_boxes = [(bx1, by1, bx2, by2) for bx1, by1, bx2, by2, ip in boxes_data if ip]
    if n > 0 and pos_boxes:
        keep = []
        for i in range(n):
            det_box = boxes[i]
            if isinstance(det_box, torch.Tensor):
                det_box = det_box.cpu().numpy()
            dx0, dy0, dx1, dy1 = det_box.flatten()[:4]
            det_cx, det_cy = (dx0 + dx1) / 2, (dy0 + dy1) / 2
            for ux1, uy1, ux2, uy2 in pos_boxes:
                if ux1 <= det_cx <= ux2 and uy1 <= det_cy <= uy2:
                    keep.append(i)
                    break
        if keep:
            masks = masks[keep]
            boxes = boxes[keep]
            scores = scores[keep]
            n = len(keep)
            print(f"[SAM3] 过滤后: {n} 个对象（中心在框内）")
        else:
            print("[SAM3] 过滤后无结果，返回全部")

    if n == 0:
        return original_image, "⚠ 未检测到对象，请调整框选范围"

    if mask_mode:
        h, w = original_image.shape[:2]
        result = masks_to_binary_image(masks, h, w)
    else:
        result = overlay_masks_on_image(original_image.copy(), masks, boxes, scores)
        result = draw_boxes_on_image(result, boxes_data)

    details = ", ".join([f"#{i}({scores[i].item():.2f})" for i in range(n)])
    print(f"[SAM3] ✓ 框选分割完成: {details}")
    _cleanup_gpu()
    return result, f"✅ 检测到 {n} 个对象: {details}"


# Box UI state management
def on_box_image_upload(image):
    if image is None:
        return None, None, [], None, "请上传图片"
    return image.copy(), image.copy(), [], None, "✅ 图片已加载，点击图片标记框的两个对角点"


def on_box_image_click(display_img, original_image, boxes_data, pending_corner, box_type, evt: gr.SelectData):
    """点击图片画框：第1次点击记录角点A，第2次点击完成框"""
    if original_image is None:
        return display_img, boxes_data, pending_corner, "⚠ 请先上传图片"
    x, y = evt.index

    if pending_corner is None:
        # 第一次点击 — 记录角点，画一个临时标记
        temp = original_image.copy()
        if boxes_data:
            temp = draw_boxes_on_image(temp, boxes_data)
        cv2.circle(temp, (int(x), int(y)), 5, (255, 165, 0), -1)
        cv2.circle(temp, (int(x), int(y)), 5, (255, 255, 255), 2)
        return temp, boxes_data, (x, y), "🔶 已标记第1个角点，请点击对角点完成框"
    else:
        # 第二次点击 — 完成框
        ax, ay = pending_corner
        x1, y1 = min(ax, x), min(ay, y)
        x2, y2 = max(ax, x), max(ay, y)
        if x2 - x1 < 3 or y2 - y1 < 3:
            return display_img, boxes_data, None, "⚠ 框太小，请重新点击"
        is_pos = box_type == "正向框（目标）"
        boxes_data = list(boxes_data) + [(x1, y1, x2, y2, is_pos)]
        annotated = draw_boxes_on_image(original_image.copy(), boxes_data)
        pos = sum(1 for b in boxes_data if b[4])
        neg = sum(1 for b in boxes_data if not b[4])
        return annotated, boxes_data, None, f"已标记 {len(boxes_data)} 个框（正向: {pos}, 负向: {neg}）"


def clear_boxes(original_image):
    if original_image is None:
        return None, [], None, "请先上传图片"
    return original_image.copy(), [], None, "✅ 已清除所有框"


# ============================================================
# Tab 3 — 视频目标跟踪（文本/点击/框选，SAM3/3.1）
# ============================================================

def _start_video_session(predictor, video_path):
    """启动视频会话，返回 (session_id, frames, fps)"""
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
        raise ValueError("无法读取视频帧")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        response = predictor.handle_request({
            "type": "start_session",
            "resource_path": video_path,
        })
    return response["session_id"], frames, fps


def _propagate_and_render(predictor, session_id, frames, fps, first_output,
                          progress, progress_start=0.15, mask_mode=False,
                          prompt_frame_idx=0, start_frame_index=None):
    """传播跟踪并渲染输出视频
    start_frame_index: 传给 propagate_in_video，对纯点击跟踪（SAM3.1）可绕过 previous_stages_out 空检查
    """
    total = len(frames)
    outputs_per_frame = {prompt_frame_idx: first_output}

    propagate_req = {
        "type": "propagate_in_video",
        "session_id": session_id,
    }
    if start_frame_index is not None:
        propagate_req["start_frame_index"] = start_frame_index

    with torch.autocast("cuda", dtype=torch.bfloat16):
        for resp in predictor.handle_stream_request(propagate_req):
            fidx = resp["frame_index"]
            outputs_per_frame[fidx] = resp["outputs"]
        pct = progress_start + 0.50 * (len(outputs_per_frame) / total)
        progress(min(pct, 0.65), desc=f"跟踪: {len(outputs_per_frame)}/{total}")

    # 渲染
    progress(0.68, desc="正在渲染输出视频...")
    timestamp = int(time.time())
    output_path = os.path.join(OUTPUT_DIR, f"track_{timestamp}.mp4")
    temp_path = os.path.join(OUTPUT_DIR, f"track_{timestamp}_tmp.mp4")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))

    for i in range(total):
        frame = frames[i]
        if i in outputs_per_frame:
            if mask_mode:
                rendered = video_masks_to_binary_frame(outputs_per_frame[i], h, w)
            else:
                rendered = overlay_video_masks(frame, outputs_per_frame[i])
        else:
            rendered = np.zeros((h, w, 3), dtype=np.uint8) if mask_mode else frame
        writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        if i % max(total // 20, 1) == 0:
            progress(0.68 + 0.25 * (i / total), desc=f"渲染: {i}/{total}")
    writer.release()
    del outputs_per_frame  # 释放 GPU mask 引用

    # H.264 重编码
    progress(0.94, desc="正在编码为 H.264...")
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

    return output_path, None


def _get_tracker(version, use_fa3=True):
    """从已加载的 predictor 中提取底层 tracker（SAM2 兼容 API）"""
    predictor = get_video_predictor(version, use_fa3=use_fa3)
    model = predictor.model
    tracker = model.tracker
    # SAM3 需要共享 backbone；SAM3.1 的 tracker 已自带
    if hasattr(model, 'detector') and hasattr(model.detector, 'backbone'):
        tracker.backbone = model.detector.backbone
    return tracker


def _tracker_propagate_and_render(tracker, inference_state, frames, fps,
                                  progress, progress_start=0.15, mask_mode=False):
    """使用底层 tracker API 传播并渲染"""
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
            # video_res shape: [n_objs, H, W]
            masks_per_frame[fidx] = (obj_ids, video_res)
            pct = progress_start + 0.50 * (len(masks_per_frame) / total)
            progress(min(pct, 0.65), desc=f"跟踪: {len(masks_per_frame)}/{total}")

    # 渲染
    progress(0.68, desc="正在渲染输出视频...")
    timestamp = int(time.time())
    output_path = os.path.join(OUTPUT_DIR, f"track_{timestamp}.mp4")
    temp_path = os.path.join(OUTPUT_DIR, f"track_{timestamp}_tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))

    for i in range(total):
        frame = frames[i]
        if i in masks_per_frame:
            obj_ids, video_res = masks_per_frame[i]
            if mask_mode:
                # 直接从 video_res 生成二值帧
                combined = np.zeros((h, w), dtype=np.uint8)
                for j in range(video_res.shape[0]):
                    m = video_res[j].cpu().numpy().squeeze()
                    if m.shape != (h, w):
                        m = cv2.resize(m.astype(np.float32), (w, h),
                                       interpolation=cv2.INTER_NEAREST)
                    combined[m > 0.0] = 255
                rendered = np.stack([combined] * 3, axis=-1)
            else:
                binary_masks = (video_res > 0.0).cpu()
                outputs = {
                    "out_obj_ids": obj_ids if not isinstance(obj_ids, list) else obj_ids,
                    "out_binary_masks": binary_masks,
                }
                rendered = overlay_video_masks(frame, outputs)
        else:
            rendered = np.zeros((h, w, 3), dtype=np.uint8) if mask_mode else frame
        writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        if i % max(total // 20, 1) == 0:
            progress(0.68 + 0.25 * (i / total), desc=f"渲染: {i}/{total}")
    writer.release()
    del masks_per_frame  # 释放 GPU mask 引用

    # H.264 重编码
    progress(0.94, desc="正在编码为 H.264...")
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

    return output_path


def track_video_text(video_path, text_prompt, model_version, mask_mode=False,
                     use_fa3=True, use_mmgp=False, mmgp_profile=4, progress=gr.Progress()):
    """使用文本提示跟踪视频中的对象"""
    print(f"\n[SAM3] === 视频文本跟踪: prompt='{text_prompt}', model={model_version}, mask={mask_mode}, fa={use_fa3} ===")
    if video_path is None:
        gr.Warning("请先上传视频")
        return None, "⚠ 请先上传视频"
    if not text_prompt or not text_prompt.strip():
        gr.Warning("请输入文本提示")
        return None, "⚠ 请输入文本提示"

    if model_version not in _video_predictors:
        return None, "⚠ 视频模型未加载，请先点击「加载视频模型」按钮"

    _set_mmgp_config(use_mmgp, mmgp_profile)
    _ensure_mode(f"video_{model_version}")
    try:
        predictor = get_video_predictor(model_version, use_fa3=use_fa3)
    except Exception as e:
        return None, f"模型加载失败: {e}"

    progress(0, desc="正在读取视频...")
    try:
        session_id, frames, fps = _start_video_session(predictor, video_path)
    except Exception as e:
        return None, f"启动会话失败: {e}"

    total = len(frames)
    h0, w0 = frames[0].shape[:2]
    print(f"[SAM3] 视频: {total} 帧, {w0}x{h0}, {fps:.1f}fps")
    progress(0.1, desc="正在检测目标...")

    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            response = predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": text_prompt.strip(),
            })
    except Exception as e:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
        return None, f"添加提示失败: {e}"

    first_output = response.get("outputs", {})
    obj_ids = first_output.get("out_obj_ids", [])
    n_objects = len(obj_ids) if hasattr(obj_ids, '__len__') else 0
    print(f"[SAM3] 首帧检测: {n_objects} 个对象")
    if n_objects == 0:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
        return None, f"⚠ 在首帧未检测到 \"{text_prompt}\""

    try:
        output_path, _ = _propagate_and_render(
            predictor, session_id, frames, fps, first_output, progress,
            mask_mode=mask_mode)
    except Exception as e:
        print(f"[SAM3] ✗ 跟踪出错: {e}")
        traceback.print_exc()
        predictor.handle_request({"type": "close_session", "session_id": session_id})
        return None, f"跟踪出错: {e}"

    predictor.handle_request({"type": "close_session", "session_id": session_id})
    del frames
    _cleanup_gpu()
    progress(1.0, desc="完成!")
    print(f"[SAM3] ✓ 视频跟踪完成: {output_path}")
    return output_path, f"✅ 跟踪完成 ({model_version}): {n_objects} 个对象, {total} 帧"


def get_first_frame(video_path):
    """提取视频首帧用于标注"""
    if video_path is None:
        return None, None, 0, "请先上传视频"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, None, 0, "⚠ 无法读取视频首帧"
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb.copy(), rgb.copy(), 0, "✅ 首帧已提取，请在图上标记点/框"


def get_video_total_frames(video_path):
    """获取视频总帧数，初始化帧选择器"""
    if video_path is None:
        return gr.update(maximum=1, value=0, visible=False), None, gr.update(visible=False), "请先上传视频"
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    cap.release()
    if not ret or total <= 0:
        return gr.update(maximum=1, value=0, visible=False), None, gr.update(visible=False), "⚠ 无法读取视频"
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return gr.update(maximum=total - 1, value=0, visible=True), rgb, gr.update(visible=True), f"共 {total} 帧，拖动滑条预览，点击「确认选帧」使用"


def preview_frame(video_path, frame_idx):
    """根据滑条位置预览指定帧"""
    if video_path is None:
        return None, "请先上传视频"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, f"⚠ 无法读取第 {int(frame_idx)} 帧"
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb, f"预览: 第 {int(frame_idx)} 帧"


def confirm_frame_selection(preview_img, frame_idx):
    """确认选帧，将预览帧作为标注用帧"""
    if preview_img is None:
        return None, None, 0, "⚠ 请先预览帧"
    idx = int(frame_idx)
    return preview_img.copy(), preview_img.copy(), idx, f"✅ 第 {idx} 帧已选定，请在图上标记点/框"


def track_video_points(video_path, points, model_version, mask_mode=False,
                       use_fa3=True, use_mmgp=False, mmgp_profile=4,
                       start_frame_idx=0, progress=gr.Progress()):
    """使用点击提示跟踪视频。SAM3 走底层 tracker API；SAM3.1 走高层 predictor API。"""
    start_frame_idx = int(start_frame_idx)
    print(f"\n[SAM3] === 视频点击跟踪: {len(points) if points else 0} 点, model={model_version}, mask={mask_mode}, fa={use_fa3}, start_frame={start_frame_idx} ===")
    if video_path is None:
        return None, "⚠ 请先上传视频"
    if not points:
        return None, "⚠ 请先在首帧上标记至少一个点"

    if model_version not in _video_predictors:
        return None, "⚠ 视频模型未加载，请先点击「加载视频模型」按钮"

    _set_mmgp_config(use_mmgp, mmgp_profile)
    _ensure_mode(f"video_{model_version}")

    if model_version == "sam3.1":
        # SAM3.1 使用高层 predictor API（init_state 接受 resource_path，不接受 video_path）
        try:
            predictor = get_video_predictor(model_version, use_fa3=use_fa3)
        except Exception as e:
            return None, f"模型加载失败: {e}"

        progress(0, desc="正在读取视频...")
        try:
            session_id, frames, fps = _start_video_session(predictor, video_path)
        except Exception as e:
            return None, f"启动会话失败: {e}"

        h, w = frames[0].shape[:2]
        total = len(frames)
        print(f"[SAM3] 视频: {total} 帧, {w}x{h} (SAM3.1 高层 API)")
        progress(0.1, desc="正在添加点提示...")

        pts_norm = [[x / w, y / h] for x, y, _ in points]
        lbls = [int(l) for _, _, l in points]
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                response = predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": start_frame_idx,
                    "points": pts_norm,
                    "point_labels": lbls,
                    "obj_id": 1,
                })
        except Exception as e:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
            traceback.print_exc()
            return None, f"添加提示失败: {e}"

        first_output = response.get("outputs") or {}
        print(f"[SAM3] 第{start_frame_idx}帧点提示完成")

        progress(0.15, desc="正在传播跟踪...")
        try:
            output_path, _ = _propagate_and_render(
                predictor, session_id, frames, fps, first_output, progress,
                mask_mode=mask_mode, prompt_frame_idx=start_frame_idx,
                start_frame_index=start_frame_idx)
        except Exception as e:
            traceback.print_exc()
            predictor.handle_request({"type": "close_session", "session_id": session_id})
            return None, f"跟踪出错: {e}"

        predictor.handle_request({"type": "close_session", "session_id": session_id})
        progress(1.0, desc="完成!")
        del frames
        _cleanup_gpu()
        return output_path, f"✅ 点击跟踪完成 ({model_version}): {total} 帧"

    else:
        # SAM3 使用底层 tracker API
        try:
            tracker = _get_tracker(model_version, use_fa3=use_fa3)
        except Exception as e:
            return None, f"模型加载失败: {e}"

        progress(0, desc="正在初始化视频...")
        try:
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
                return None, "⚠ 无法读取视频帧"

            h, w = frames[0].shape[:2]
            total = len(frames)
            print(f"[SAM3] 视频: {total} 帧, {w}x{h}")

            with torch.autocast("cuda", dtype=torch.bfloat16):
                inference_state = tracker.init_state(video_path=video_path)
            # MMGP 会把参数卸载到 CPU，导致 tracker.device 返回 cpu；
            # 强制覆盖为 cuda，确保 add_new_points_or_box 内部把张量移到正确设备
            inference_state["device"] = torch.device("cuda")
        except Exception as e:
            return None, f"初始化失败: {e}"

        progress(0.1, desc="正在添加点提示...")
        pts_tensor = torch.tensor([[x / w, y / h] for x, y, _ in points], dtype=torch.float32).cuda()
        lbl_tensor = torch.tensor([l for _, _, l in points], dtype=torch.int32).cuda()

        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, out_obj_ids, low_res, video_res = tracker.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=start_frame_idx,
                    obj_id=1,
                    points=pts_tensor,
                    labels=lbl_tensor,
                    clear_old_points=False,
                )
            print(f"[SAM3] 第{start_frame_idx}帧点提示: obj_ids={out_obj_ids}")
        except Exception as e:
            traceback.print_exc()
            return None, f"添加提示失败: {e}"

        progress(0.15, desc="正在传播跟踪...")
        try:
            output_path = _tracker_propagate_and_render(
                tracker, inference_state, frames, fps, progress,
                mask_mode=mask_mode)
        except Exception as e:
            traceback.print_exc()
            return None, f"跟踪出错: {e}"

        progress(1.0, desc="完成!")
        del inference_state, frames
        _cleanup_gpu()
        return output_path, f"✅ 点击跟踪完成 ({model_version}): {total} 帧"


def track_video_box(video_path, vid_box_data, model_version, mask_mode=False,
                    text_prompt="", use_fa3=True, use_mmgp=False, mmgp_profile=4,
                    start_frame_idx=0, progress=gr.Progress()):
    """使用框选提示跟踪视频。支持多框、正向框/负向框。
    SAM3.1（任意框） / SAM3+有文本 / 多框 / 含负向框 → 高层 predictor API；
    SAM3+无文本+单正向框 → 底层 tracker API（add_new_points_or_box）。
    """
    start_frame_idx = int(start_frame_idx)
    if video_path is None:
        return None, "⚠ 请先上传视频"
    if not vid_box_data:
        return None, "⚠ 请先在首帧上画框"

    # 兼容旧数据格式（4 元组无标签）：默认视为正向框
    norm_boxes = []
    for b in vid_box_data:
        if len(b) >= 5:
            x1, y1, x2, y2, is_pos = b[0], b[1], b[2], b[3], bool(b[4])
        else:
            x1, y1, x2, y2 = b[:4]
            is_pos = True
        norm_boxes.append((x1, y1, x2, y2, is_pos))

    pos_boxes = [b for b in norm_boxes if b[4]]
    neg_boxes = [b for b in norm_boxes if not b[4]]
    if not pos_boxes:
        return None, "⚠ 至少需要 1 个正向框"

    text_prompt = (text_prompt or "").strip()
    print(f"\n[SAM3] === 视频框选跟踪: 正向框={len(pos_boxes)}, 负向框={len(neg_boxes)}, "
          f"text='{text_prompt}', model={model_version}, mask={mask_mode} ===")

    if model_version not in _video_predictors:
        return None, "⚠ 视频模型未加载，请先点击「加载视频模型」按钮"

    _set_mmgp_config(use_mmgp, mmgp_profile)
    _ensure_mode(f"video_{model_version}")

    # 任何会让底层 tracker 走不通的情况都升级到高层 API
    use_high_level = (
        text_prompt
        or model_version == "sam3.1"
        or len(norm_boxes) > 1
        or len(neg_boxes) > 0
    )

    if use_high_level:
        # ---- 高层 handle_request API（SAM3.1 / 有文本 / 多框 / 含负向框） ----
        try:
            predictor = get_video_predictor(model_version, use_fa3=use_fa3)
        except Exception as e:
            return None, f"模型加载失败: {e}"

        progress(0, desc="正在读取视频...")
        try:
            session_id, frames, fps = _start_video_session(predictor, video_path)
        except Exception as e:
            return None, f"启动会话失败: {e}"

        h, w = frames[0].shape[:2]
        total = len(frames)
        prompt_desc_parts = []
        if text_prompt:
            prompt_desc_parts.append("文本")
        prompt_desc_parts.append(f"{len(pos_boxes)}正")
        if neg_boxes:
            prompt_desc_parts.append(f"{len(neg_boxes)}负")
        prompt_desc = "+".join(prompt_desc_parts)
        print(f"[SAM3] 视频: {total} 帧, {w}x{h}, {fps:.1f}fps (高层 API, {prompt_desc})")
        progress(0.1, desc=f"正在添加{prompt_desc}提示...")

        # 归一化框坐标 → xywh 格式 [xmin, ymin, width, height] 0~1
        bounding_boxes = []
        bounding_box_labels = []
        for x1, y1, x2, y2, is_pos in norm_boxes:
            bounding_boxes.append([x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h])
            bounding_box_labels.append(1 if is_pos else 0)

        # SAM3 视频 predictor 的初始 visual prompt 一次只允许 1 个 box（见
        # sam3_video_inference.py::_get_visual_prompt），无论是否有文本都受此限制。
        # 多框时分步调用：第 1 框作为初始 visual prompt（带文本一起），
        # 之后的框对同一 obj_id 做 refinement（必须 clear_old_boxes=False，否则
        # 第一次的框会被清掉；并显式指定 obj_id，否则会被当作新对象）。
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # 第 1 个框（含文本，如果有）
                first_req = {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": start_frame_idx,
                    "bounding_boxes": [bounding_boxes[0]],
                    "bounding_box_labels": [bounding_box_labels[0]],
                }
                if text_prompt:
                    first_req["text"] = text_prompt
                response = predictor.handle_request(first_req)

                # 取第一次返回的 obj_id 用于后续 refinement
                first_outputs = response.get("outputs") or {}
                first_obj_ids = first_outputs.get("out_obj_ids", [])
                if hasattr(first_obj_ids, "tolist"):
                    first_obj_ids = first_obj_ids.tolist()
                refine_obj_id = int(first_obj_ids[0]) if len(first_obj_ids) > 0 else 1

                # 后续框逐个作为 refinement prompt 加到同一对象上
                for i in range(1, len(bounding_boxes)):
                    response = predictor.handle_request({
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": start_frame_idx,
                        "bounding_boxes": [bounding_boxes[i]],
                        "bounding_box_labels": [bounding_box_labels[i]],
                        "obj_id": refine_obj_id,
                        "clear_old_boxes": False,
                    })
        except Exception as e:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
            traceback.print_exc()
            return None, f"添加提示失败: {e}"

        first_output = response.get("outputs") or {}
        obj_ids = first_output.get("out_obj_ids", [])
        n_objects = len(obj_ids) if hasattr(obj_ids, '__len__') else 0
        print(f"[SAM3] 第{start_frame_idx}帧检测: {n_objects} 个对象")
        if n_objects == 0:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
            return None, "⚠ 在指定帧未检测到目标"

        progress(0.15, desc="正在传播跟踪...")
        try:
            output_path, _ = _propagate_and_render(
                predictor, session_id, frames, fps, first_output, progress,
                mask_mode=mask_mode, prompt_frame_idx=start_frame_idx)
        except Exception as e:
            traceback.print_exc()
            predictor.handle_request({"type": "close_session", "session_id": session_id})
            return None, f"跟踪出错: {e}"

        predictor.handle_request({"type": "close_session", "session_id": session_id})
        progress(1.0, desc="完成!")
        del frames
        _cleanup_gpu()
        suffix = f"（{prompt_desc}）"
        if text_prompt:
            suffix += f" 文本: '{text_prompt}'"
        return output_path, f"✅ 框选跟踪完成 ({model_version}): {n_objects} 个对象, {total} 帧 {suffix}"

    else:
        # ---- SAM3 + 无文本 + 单正向框：底层 tracker API（add_new_points_or_box） ----
        box_x1, box_y1, box_x2, box_y2 = pos_boxes[0][:4]
        try:
            tracker = _get_tracker(model_version, use_fa3=use_fa3)
        except Exception as e:
            return None, f"模型加载失败: {e}"

        progress(0, desc="正在初始化视频...")
        try:
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
                return None, "⚠ 无法读取视频帧"

            h, w = frames[0].shape[:2]
            total = len(frames)
            print(f"[SAM3] 视频: {total} 帧, {w}x{h} (SAM3 底层 tracker API)")

            with torch.autocast("cuda", dtype=torch.bfloat16):
                inference_state = tracker.init_state(video_path=video_path)
            # MMGP 会把参数卸载到 CPU，导致 tracker.device 返回 cpu；强制覆盖为 cuda
            inference_state["device"] = torch.device("cuda")
        except Exception as e:
            return None, f"初始化失败: {e}"

        progress(0.1, desc="正在添加框提示...")
        # tracker API 用归一化 xyxy: [x_min/W, y_min/H, x_max/W, y_max/H]
        # 不提前 .cuda()：内部还会创建 CPU 零点张量与 box_coords cat，留给
        # inference_state["device"]="cuda" 统一在 add_new_points_or_box 末尾搬运
        box_tensor = torch.tensor(
            [[box_x1 / w, box_y1 / h, box_x2 / w, box_y2 / h]],
            dtype=torch.float32,
        )

        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, out_obj_ids, low_res, video_res = tracker.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=start_frame_idx,
                    obj_id=1,
                    box=box_tensor,
                )
            print(f"[SAM3] 第{start_frame_idx}帧框提示: obj_ids={out_obj_ids}")
        except Exception as e:
            traceback.print_exc()
            return None, f"添加提示失败: {e}"

        progress(0.15, desc="正在传播跟踪...")
        try:
            output_path = _tracker_propagate_and_render(
                tracker, inference_state, frames, fps, progress,
                mask_mode=mask_mode)
        except Exception as e:
            traceback.print_exc()
            return None, f"跟踪出错: {e}"

        progress(1.0, desc="完成!")
        del inference_state, frames
        _cleanup_gpu()
        return output_path, f"✅ 框选跟踪完成 ({model_version}): {total} 帧"


# Video point UI helpers
def on_video_frame_click(display_img, original_frame, vid_points, point_type, evt: gr.SelectData):
    if original_frame is None:
        return display_img, vid_points, "⚠ 请先提取首帧"
    x, y = evt.index
    label = 1 if point_type == "正向点（前景）" else 0
    vid_points = list(vid_points) + [(x, y, label)]
    annotated = draw_points_on_image(original_frame.copy(), vid_points)
    pos = sum(1 for _, _, l in vid_points if l == 1)
    neg = sum(1 for _, _, l in vid_points if l == 0)
    return annotated, vid_points, f"已标记 {len(vid_points)} 个点（正向: {pos}, 负向: {neg}）"


def clear_video_points(original_frame):
    if original_frame is None:
        return None, [], "请先提取首帧"
    return original_frame.copy(), [], "✅ 已清除标记"


# Video box UI helpers
def on_video_box_click(display_img, original_frame, vid_box_data, pending_corner, box_type, evt: gr.SelectData):
    """在视频首帧上点击画框，支持正向框/负向框"""
    if original_frame is None:
        return display_img, vid_box_data, pending_corner, "⚠ 请先提取首帧"
    x, y = evt.index
    if pending_corner is None:
        temp = original_frame.copy()
        if vid_box_data:
            temp = draw_boxes_on_image(temp, vid_box_data)
        cv2.circle(temp, (int(x), int(y)), 5, (255, 165, 0), -1)
        cv2.circle(temp, (int(x), int(y)), 5, (255, 255, 255), 2)
        return temp, vid_box_data, (x, y), "🔶 已标记第1个角点，请点击对角点完成框"
    else:
        ax, ay = pending_corner
        x1, y1 = min(ax, x), min(ay, y)
        x2, y2 = max(ax, x), max(ay, y)
        if x2 - x1 < 3 or y2 - y1 < 3:
            return display_img, vid_box_data, None, "⚠ 框太小，请重新点击"
        is_pos = box_type == "正向框（目标）"
        vid_box_data = list(vid_box_data) + [(x1, y1, x2, y2, is_pos)]
        annotated = draw_boxes_on_image(original_frame.copy(), vid_box_data)
        pos = sum(1 for b in vid_box_data if b[4])
        neg = sum(1 for b in vid_box_data if not b[4])
        return annotated, vid_box_data, None, f"已标记 {len(vid_box_data)} 个框（正向: {pos}, 负向: {neg}）"



def clear_video_boxes(original_frame):
    if original_frame is None:
        return None, [], None, "请先提取首帧"
    return original_frame.copy(), [], None, "✅ 已清除框"


# ============================================================
# Tab 4 — 交互式点击分割
# ============================================================

def on_image_upload(image):
    if image is None:
        return None, None, [], "请上传图片"
    return image.copy(), image.copy(), [], "✅ 图片已加载，点击图片添加标记点"


def on_image_click(display_img, original_image, points, point_type, evt: gr.SelectData):
    if original_image is None:
        return display_img, points, "⚠ 请先上传图片"
    x, y = evt.index
    label = 1 if point_type == "正向点（前景）" else 0
    points = list(points) + [(x, y, label)]
    annotated = draw_points_on_image(original_image.copy(), points)
    pos = sum(1 for _, _, l in points if l == 1)
    neg = sum(1 for _, _, l in points if l == 0)
    return annotated, points, f"已标记 {len(points)} 个点（正向: {pos}, 负向: {neg}）"


def clear_points(original_image):
    if original_image is None:
        return None, [], "请先上传图片"
    return original_image.copy(), [], "✅ 已清除所有标记点"


def segment_with_points(original_image, points, mask_mode=False, model_version="sam3", use_mmgp=False, mmgp_profile=4):
    pos = sum(1 for _, _, l in points if l == 1) if points else 0
    neg = sum(1 for _, _, l in points if l == 0) if points else 0
    print(f"\n[SAM3] === 点击分割请求: {len(points) if points else 0} 个点 (正:{pos}, 负:{neg}), mask={mask_mode}, ver={model_version} ===")
    if original_image is None:
        gr.Warning("请先上传图片")
        return None, "⚠ 请先上传图片"
    if not points:
        gr.Warning("请先点击图片标记至少一个点")
        return None, "⚠ 请先标记至少一个点"

    if _interactive_model is None:
        return None, "⚠ 图像模型未加载，请先点击「加载图像模型」按钮"

    _set_mmgp_config(use_mmgp, mmgp_profile)
    _ensure_mode(f"interactive_{model_version}")
    try:
        model, processor = get_interactive_model(model_version)
    except Exception as e:
        return None, f"模型加载失败: {e}"

    pil_img = Image.fromarray(original_image)

    try:
        t0 = time.time()
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            state = processor.set_image(pil_img)
            point_coords = np.array([[x, y] for x, y, _ in points])
            point_labels = np.array([l for _, _, l in points])
            masks, scores, logits = model.predict_inst(
                state,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
        print(f"[SAM3] 推理耗时: {time.time() - t0:.2f}s, 返回 {len(scores)} 个掩码")
    except Exception as e:
        print(f"[SAM3] ✗ 点分割出错: {e}")
        traceback.print_exc()
        return None, f"分割出错: {e}"

    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    best_score = scores[best_idx]

    if mask_mode:
        h, w = original_image.shape[:2]
        mask_bool = best_mask > 0.5
        binary = np.zeros((h, w), dtype=np.uint8)
        binary[mask_bool] = 255
        result = np.stack([binary] * 3, axis=-1)
    else:
        overlay = original_image.copy()
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
            mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 2)
        result = draw_points_on_image(overlay, points)

    all_scores = ", ".join([f"mask{i}={s:.3f}" for i, s in enumerate(scores)])
    print(f"[SAM3] ✓ 点击分割完成: {all_scores}")
    _cleanup_gpu()
    return result, f"✅ 分割完成 (得分: {best_score:.3f}，共 {len(scores)} 个候选)"


# ============================================================
# Tab 5 — 批量图像分割
# ============================================================
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

def _collect_images_from_folder(folder_path):
    """从文件夹收集所有图片文件路径，按文件名排序"""
    folder = folder_path.strip().strip('"').strip("'")
    if not os.path.isdir(folder):
        raise ValueError(f"文件夹不存在: {folder}")
    files = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ])
    if not files:
        raise ValueError(f"文件夹中未找到图片文件: {folder}")
    return files


def _extract_frames_from_video(video_path, max_frames=0):
    """从视频提取所有帧（或最多 max_frames 帧），返回 PIL Image 列表和帧号"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames > 0 and idx >= max_frames:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((idx, Image.fromarray(rgb)))
        idx += 1
    cap.release()
    if not frames:
        raise ValueError("视频中未读取到任何帧")
    return frames


def batch_segment(source_mode, folder_path, video_file, text_prompt, confidence,
                  frame_interval, mask_mode=False, model_version="sam3",
                  use_mmgp=False, mmgp_profile=4, progress=gr.Progress()):
    """批量图像分割：支持文件夹模式和视频拆帧模式"""
    print(f"\n[SAM3] === 批量分割: mode={source_mode}, prompt='{text_prompt}', conf={confidence}, mask={mask_mode}, ver={model_version} ===")
    if not text_prompt or not text_prompt.strip():
        return [], None, "⚠ 请输入文本提示"
    if _image_processor is None:
        return [], None, "⚠ 图像模型未加载，请先点击「加载图像模型」按钮"

    _set_mmgp_config(use_mmgp, mmgp_profile)
    _ensure_mode(f"image_{model_version}")

    # 收集图像
    progress(0, desc="正在收集图像...")
    pil_images = []   # (label, PIL.Image)
    video_fps = None
    video_total_frames = 0
    try:
        if source_mode == "图片文件夹":
            if not folder_path or not folder_path.strip():
                return [], None, "⚠ 请输入文件夹路径"
            files = _collect_images_from_folder(folder_path)
            for f in files:
                pil_images.append((os.path.basename(f), Image.open(f).convert("RGB")))
        else:  # 视频拆帧
            if video_file is None:
                return [], None, "⚠ 请上传视频"
            interval = max(1, int(frame_interval))
            # 获取原始视频参数
            cap = cv2.VideoCapture(video_file)
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            all_frames = _extract_frames_from_video(video_file)
            for idx, img in all_frames:
                if idx % interval == 0:
                    pil_images.append((f"frame_{idx:06d}", img))
    except Exception as e:
        return [], None, f"⚠ 收集图像失败: {e}"

    total = len(pil_images)
    if total == 0:
        return [], None, "⚠ 未收集到任何图像"
    print(f"[SAM3] 共 {total} 张图像待处理")

    # 加载模型
    progress(0.02, desc="正在加载模型...")
    try:
        processor = get_image_processor(model_version)
    except Exception as e:
        return [], None, f"模型加载失败: {e}"

    processor.confidence_threshold = confidence

    # 逐张推理
    gallery_items = []
    result_frames = []  # 用于视频拼接
    output_subdir = os.path.join(OUTPUT_DIR, f"batch_{int(time.time())}")
    os.makedirs(output_subdir, exist_ok=True)

    t0 = time.time()
    for i, (label, pil_img) in enumerate(pil_images):
        progress((i + 1) / (total + 1), desc=f"推理: {i+1}/{total} — {label}")
        try:
            with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
                state = processor.set_image(pil_img)
                state = processor.set_text_prompt(state=state, prompt=text_prompt.strip())

            masks = state["masks"]
            boxes = state["boxes"]
            scores = state["scores"]
            n = scores.shape[0] if isinstance(scores, torch.Tensor) else len(scores)

            img_np = np.array(pil_img)
            if n > 0:
                if mask_mode:
                    h_img, w_img = img_np.shape[:2]
                    result = masks_to_binary_image(masks, h_img, w_img)
                else:
                    result = overlay_masks_on_image(img_np, masks, boxes, scores)
            else:
                result = np.zeros_like(img_np) if mask_mode else img_np

            # 保存结果
            save_path = os.path.join(output_subdir, f"{label}.png")
            Image.fromarray(result).save(save_path)
            gallery_items.append((save_path, f"{label} ({n}个对象)"))
            result_frames.append(result)

        except Exception as e:
            print(f"[SAM3] ✗ 处理 {label} 失败: {e}")
            traceback.print_exc()
            gallery_items.append((np.array(pil_img), f"{label} (失败)"))
            result_frames.append(np.array(pil_img))

    elapsed = time.time() - t0
    avg = elapsed / total if total > 0 else 0

    # 视频拆帧模式：将结果帧拼回视频
    output_video = None
    if source_mode == "视频拆帧" and result_frames and video_fps:
        progress(0.92, desc="正在合成结果视频...")
        # 输出帧率 = 原始帧率 / 抽帧间隔
        out_fps = video_fps / max(1, int(frame_interval))
        h, w = result_frames[0].shape[:2]
        temp_path = os.path.join(output_subdir, "result_tmp.mp4")
        final_path = os.path.join(output_subdir, "result_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_path, fourcc, out_fps, (w, h))
        for frame in result_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

        # H.264 重编码
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_path,
                 "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                 "-pix_fmt", "yuv420p", final_path],
                check=True, capture_output=True,
            )
            os.remove(temp_path)
        except Exception:
            if os.path.exists(temp_path):
                os.replace(temp_path, final_path)
        output_video = final_path
        print(f"[SAM3] 结果视频已保存: {final_path} ({out_fps:.1f}fps)")

    msg = f"✅ 批量分割完成: {total} 张, 耗时 {elapsed:.1f}s (平均 {avg:.2f}s/张)\n结果保存至: {output_subdir}"
    if output_video:
        msg += f"\n结果视频: {os.path.basename(output_video)}"
    print(f"[SAM3] ✓ {msg}")
    del result_frames
    _cleanup_gpu()
    progress(1.0, desc="完成!")
    return gallery_items, output_video, msg


# ============================================================
# 模型显式加载（供 UI 按钮调用）
# ============================================================

def load_image_model(model_version, use_mmgp, mmgp_profile):
    """显式加载/重载图像模型。强制卸载旧模型使 mmgp 设置立即生效。"""
    global _image_processor, _interactive_model, _interactive_processor, _video_predictors, _active_mode
    print(f"\n[SAM3] === 显式加载图像模型: ver={model_version}, mmgp={use_mmgp}, profile={mmgp_profile} ===")
    # 强制卸载旧图像/交互式模型，确保新 mmgp 设置干净生效
    if _image_processor is not None:
        print("[SAM3] 卸载旧图像分割模型...")
        del _image_processor
        _image_processor = None
        _release_mmgp_for("image.")
    if _interactive_model is not None:
        print("[SAM3] 卸载旧交互式分割模型...")
        del _interactive_model, _interactive_processor
        _interactive_model = None
        _interactive_processor = None
        _release_mmgp_for("interactive.")
    # 同时卸载视频模型（显存只够一侧）
    for ver in list(_video_predictors.keys()):
        print(f"[SAM3] 卸载视频模型 ({ver}) 以释放显存...")
        del _video_predictors[ver]
        _release_mmgp_for(f"video.{ver}.")
    _video_predictors.clear()
    _active_mode = None
    _cleanup_gpu()

    _set_mmgp_config(use_mmgp, mmgp_profile)
    try:
        _ensure_mode(f"image_{model_version}")
        get_image_processor(model_version)
        # 交互式点击分割仅 sam3 支持（sam3.1_multiplex.pt 不含相关权重）
        if model_version == "sam3":
            get_interactive_model("sam3")
        mmgp_note = f"（mmgp profile={mmgp_profile}）" if use_mmgp else "（未启用 mmgp）"
        img_status = f"✅ 图像模型 ({model_version}) 加载完成 {mmgp_note}"
        vid_status = "⚠ 视频模型已卸载（加载图像模型时释放显存）——如需视频推理请重新加载"
        return img_status, vid_status
    except Exception as e:
        traceback.print_exc()
        err = f"❌ 图像模型加载失败: {e}"
        return err, ""


def load_video_model(model_version, use_fa3, use_mmgp, mmgp_profile, sam31_batch_size=16):
    """显式加载/重载视频模型。强制卸载旧模型使 mmgp 设置立即生效。"""
    global _image_processor, _interactive_model, _interactive_processor, _video_predictors, _video_use_fa, _active_mode
    print(f"\n[SAM3] === 显式加载视频模型: ver={model_version}, fa={use_fa3}, mmgp={use_mmgp}, profile={mmgp_profile}, sam31_batch={sam31_batch_size} ===")
    # 强制卸载所有已缓存的视频预测器
    for ver in list(_video_predictors.keys()):
        print(f"[SAM3] 卸载旧视频模型 ({ver})...")
        del _video_predictors[ver]
        _release_mmgp_for(f"video.{ver}.")
    _video_predictors.clear()
    _video_use_fa = use_fa3
    # 同时卸载图像/交互式模型（显存只够一侧）
    if _image_processor is not None:
        print("[SAM3] 卸载图像分割模型以释放显存...")
        del _image_processor
        _image_processor = None
        _release_mmgp_for("image.")
    if _interactive_model is not None:
        print("[SAM3] 卸载交互式分割模型以释放显存...")
        del _interactive_model, _interactive_processor
        _interactive_model = None
        _interactive_processor = None
        _release_mmgp_for("interactive.")
    _active_mode = None
    _cleanup_gpu()

    _set_mmgp_config(use_mmgp, mmgp_profile)
    try:
        _ensure_mode(f"video_{model_version}")
        predictor = get_video_predictor(model_version, use_fa3)
        # 加载完成后应用 batch 参数（mmgp 路径已在 get_video_predictor 内调用过
        # _apply_mmgp_to_video_predictor，这里再次覆盖以应用 UI 设置的 batch_size）
        if use_mmgp and _mmgp_enabled and predictor is not None and model_version == "sam3.1":
            _m = getattr(predictor, "model", None)
            if _m is not None:
                _eff = int(sam31_batch_size)
                _use_batched = _eff > 1
                if hasattr(_m, "use_batched_grounding"):
                    _m.use_batched_grounding = _use_batched
                if hasattr(_m, "batched_grounding_batch_size"):
                    _m.batched_grounding_batch_size = _eff
                if hasattr(_m, "postprocess_batch_size"):
                    _m.postprocess_batch_size = _eff
                print(f"[SAM3] SAM3.1 batched_grounding_batch_size 已设为 {_eff}")
        fa_note = "FA2" if use_fa3 else "SDPA"
        batch_note = f"batch={int(sam31_batch_size)}" if model_version == "sam3.1" else ""
        mmgp_note = f"（mmgp profile={mmgp_profile}{', ' + batch_note if batch_note else ''}）" if use_mmgp else "（未启用 mmgp）"
        vid_status = f"✅ 视频模型 ({model_version}, {fa_note}) 加载完成 {mmgp_note}"
        img_status = "⚠ 图像模型已卸载（加载视频模型时释放显存）——如需图像推理请重新加载"
        return img_status, vid_status
    except Exception as e:
        traceback.print_exc()
        err = f"❌ 视频模型加载失败: {e}"
        return "", err


# ============================================================
# Gradio UI
# ============================================================

def build_ui():
    with gr.Blocks(
        title="SAM3 分割工具",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# 🎯 SAM3 智能分割工具")

        # ============ 顶部：全局模型选择 ============
        with gr.Row():
            global_model = gr.Radio(
                ["sam3", "sam3.1"], value="sam3",
                label="模型版本（SAM3.1 使用 Object Multiplex，多对象更快）",
            )
            global_fa = gr.Checkbox(
                label="⚗ Flash Attention（视频处理加速，需 flash_attn 库）",
                value=True,
            )
            global_mmgp = gr.Checkbox(
                label="🧠 启用 mmgp（显存卸载）",
                value=True,
            )
            global_mmgp_profile = gr.Slider(
                1, 4, value=4, step=1,
                label="mmgp profile",
            )
            sam31_batch_md = gr.Markdown(
                "**SAM3.1 backbone 批大小**  \nbatch=16 快但显存高；（1:约7GB, 4:约9GB, 16:约12GB）",
                visible=False,
            )
            sam31_batch_size = gr.Number(
                value=4,
                label="批大小",
                visible=False,
            )

        # ============================================================
        # 大选项卡 A：图片处理
        # ============================================================
        with gr.Tab("🖼️ 图片处理"):
            with gr.Row():
                img_load_btn = gr.Button("📥 加载图像模型", variant="secondary", scale=1)
                img_load_status = gr.Textbox(
                    label="模型状态",
                    placeholder="点击「加载图像模型」以载入/重载模型（切换 mmgp 后须重新加载）",
                    interactive=False,
                    scale=4,
                )
            with gr.Tabs():

                # ---------- 文本分割 ----------
                with gr.Tab("📝 文本分割"):
                    gr.Markdown("上传图片，输入要分割的对象名称（英文），模型会自动检测并分割。")
                    with gr.Row():
                        with gr.Column():
                            img_input = gr.Image(
                                type="numpy", label="上传图片",
                                sources=["upload", "clipboard"],
                            )
                            img_text = gr.Textbox(
                                label="文本提示",
                                placeholder="输入对象名称，如: person, car, dog ...",
                            )
                            img_conf = gr.Slider(
                                0.1, 1.0, value=0.5, step=0.05, label="置信度阈值",
                            )
                            with gr.Row():
                                img_btn = gr.Button("🔍 开始分割", variant="primary", size="lg")
                                img_mask_mode = gr.Checkbox(label="Mask模式", value=False)
                        with gr.Column():
                            img_output = gr.Image(type="numpy", label="分割结果")
                            img_status = gr.Textbox(label="状态", interactive=False)

                    img_btn.click(
                        segment_image,
                        inputs=[img_input, img_text, img_conf, img_mask_mode, global_model,
                                global_mmgp, global_mmgp_profile],
                        outputs=[img_output, img_status],
                        concurrency_limit=1,
                    )

                # ---------- 框选分割 ----------
                with gr.Tab("🔲 框选分割"):
                    gr.Markdown(
                        "在图片上**点击两次**画框（角点A → 对角点B）。\n"
                        "- 🟢 **正向框**：圈选目标  |  🔴 **负向框**：排除区域\n"
                        "- 可选填文本提示辅助检测"
                    )
                    box_original_state = gr.State(None)
                    box_data_state = gr.State([])
                    box_pending_corner = gr.State(None)

                    with gr.Row():
                        with gr.Column():
                            box_img = gr.Image(
                                type="numpy", label="上传图片（点击画框）",
                                sources=["upload", "clipboard"],
                            )
                            box_text = gr.Textbox(
                                label="文本提示（可选）",
                                placeholder="留空 = 分割框内对象；填写 = 先检测再用框精确指定",
                            )
                            box_type = gr.Radio(
                                ["正向框（目标）", "负向框（排除）"],
                                value="正向框（目标）", label="框类型",
                            )
                            box_conf = gr.Slider(
                                0.1, 1.0, value=0.5, step=0.05, label="置信度阈值",
                            )
                            with gr.Row():
                                box_seg_btn = gr.Button("🔍 分割", variant="primary", size="lg")
                                box_clear_btn = gr.Button("🗑️ 清除框", variant="secondary")
                                box_mask_mode = gr.Checkbox(label="Mask模式", value=False)
                        with gr.Column():
                            box_output = gr.Image(type="numpy", label="分割结果")
                            box_status = gr.Textbox(label="状态", interactive=False)

                    box_img.upload(
                        on_box_image_upload,
                        inputs=[box_img],
                        outputs=[box_img, box_original_state, box_data_state,
                                 box_pending_corner, box_status],
                    )
                    box_img.select(
                        on_box_image_click,
                        inputs=[box_img, box_original_state, box_data_state,
                                box_pending_corner, box_type],
                        outputs=[box_img, box_data_state, box_pending_corner, box_status],
                    )
                    box_clear_btn.click(
                        clear_boxes,
                        inputs=[box_original_state],
                        outputs=[box_img, box_data_state, box_pending_corner, box_status],
                    )
                    box_seg_btn.click(
                        segment_image_with_boxes,
                        inputs=[box_original_state, box_data_state, box_text,
                                box_conf, box_mask_mode, global_model,
                                global_mmgp, global_mmgp_profile],
                        outputs=[box_output, box_status],
                        concurrency_limit=1,
                    )

                # ---------- 点击分割 ----------
                with gr.Tab("👆 点击分割") as click_tab:
                    gr.Markdown(
                        "点击图片添加标记点：\n"
                        "- 🟢 **正向点**：标记目标区域  |  🔴 **负向点**：排除背景"
                    )
                    original_image_state = gr.State(None)
                    points_state = gr.State([])

                    with gr.Row():
                        with gr.Column():
                            click_img = gr.Image(
                                type="numpy", label="上传图片并点击标记",
                                sources=["upload", "clipboard"],
                            )
                            point_type = gr.Radio(
                                ["正向点（前景）", "负向点（背景）"],
                                value="正向点（前景）", label="标记类型",
                            )
                            with gr.Row():
                                click_seg_btn = gr.Button("🔍 分割", variant="primary")
                                click_clear_btn = gr.Button("🗑️ 清除标记", variant="secondary")
                                click_mask_mode = gr.Checkbox(label="Mask模式", value=False)
                        with gr.Column():
                            click_output = gr.Image(type="numpy", label="分割结果")
                            click_status = gr.Textbox(label="状态", interactive=False)

                    click_img.upload(
                        on_image_upload,
                        inputs=[click_img],
                        outputs=[click_img, original_image_state, points_state, click_status],
                    )
                    click_img.select(
                        on_image_click,
                        inputs=[click_img, original_image_state, points_state, point_type],
                        outputs=[click_img, points_state, click_status],
                    )
                    click_seg_btn.click(
                        segment_with_points,
                        inputs=[original_image_state, points_state, click_mask_mode, global_model,
                                global_mmgp, global_mmgp_profile],
                        outputs=[click_output, click_status],
                        concurrency_limit=1,
                    )
                    click_clear_btn.click(
                        clear_points,
                        inputs=[original_image_state],
                        outputs=[click_img, points_state, click_status],
                    )

                # ---------- 批量分割 ----------
                with gr.Tab("📦 批量分割"):
                    gr.Markdown(
                        "对多张图片执行相同的文本提示分割。\n"
                        "- **图片文件夹**：输入包含图片的文件夹路径\n"
                        "- **视频拆帧**：上传视频，自动拆帧后逐帧分割"
                    )
                    batch_source_mode = gr.Radio(
                        ["图片文件夹", "视频拆帧"], value="图片文件夹",
                        label="输入模式",
                    )
                    with gr.Row():
                        with gr.Column():
                            batch_folder = gr.Textbox(
                                label="图片文件夹路径",
                                placeholder=r"例如: D:\my_images",
                                visible=True,
                            )
                            batch_video = gr.Video(label="上传视频", visible=False)
                            batch_frame_interval = gr.Slider(
                                1, 30, value=1, step=1,
                                label="抽帧间隔（每 N 帧取 1 帧）",
                                visible=False,
                            )
                            batch_text = gr.Textbox(
                                label="文本提示（英文）",
                                placeholder="例如: person, cat, car...",
                            )
                            batch_conf = gr.Slider(
                                0.1, 1.0, value=0.5, step=0.05, label="置信度阈值",
                            )
                            with gr.Row():
                                batch_run_btn = gr.Button("🚀 开始批量分割", variant="primary")
                                batch_mask_mode = gr.Checkbox(label="Mask模式", value=False)
                        with gr.Column():
                            batch_gallery = gr.Gallery(
                                label="分割结果", columns=3, height="auto",
                                object_fit="contain",
                            )
                            batch_result_video = gr.Video(
                                label="结果视频预览（仅视频拆帧模式）",
                            )
                            batch_status = gr.Textbox(label="状态", interactive=False)

                    def toggle_batch_mode(mode):
                        is_folder = mode == "图片文件夹"
                        return (
                            gr.update(visible=is_folder),
                            gr.update(visible=not is_folder),
                            gr.update(visible=not is_folder),
                        )

                    batch_source_mode.change(
                        toggle_batch_mode,
                        inputs=[batch_source_mode],
                        outputs=[batch_folder, batch_video, batch_frame_interval],
                    )
                    batch_run_btn.click(
                        batch_segment,
                        inputs=[batch_source_mode, batch_folder, batch_video,
                                batch_text, batch_conf, batch_frame_interval,
                                batch_mask_mode, global_model,
                                global_mmgp, global_mmgp_profile],
                        outputs=[batch_gallery, batch_result_video, batch_status],
                        concurrency_limit=1,
                    )

        # ============================================================
        # 大选项卡 B：视频处理
        # ============================================================
        with gr.Tab("🎬 视频处理"):
            with gr.Row():
                vid_load_btn = gr.Button("📥 加载视频模型", variant="secondary", scale=1)
                vid_load_status = gr.Textbox(
                    label="模型状态",
                    placeholder="点击「加载视频模型」以载入/重载模型（切换 mmgp / Flash Attention 后须重新加载）",
                    interactive=False,
                    scale=4,
                )
            vid_load_btn.click(
                load_video_model,
                inputs=[global_model, global_fa, global_mmgp, global_mmgp_profile, sam31_batch_size],
                outputs=[img_load_status, vid_load_status],
                concurrency_limit=1,
            )
            with gr.Tabs():

                # ---------- 文本跟踪 ----------
                with gr.Tab("📝 文本跟踪"):
                    with gr.Row():
                        with gr.Column():
                            vid_text_input = gr.Video(
                                label="上传视频", sources=["upload"],
                            )
                            vid_text_prompt = gr.Textbox(
                                label="文本提示",
                                placeholder="输入对象名称，如: person, ball, car ...",
                            )
                            with gr.Row():
                                vid_text_btn = gr.Button(
                                    "🎯 开始跟踪", variant="primary", size="lg",
                                )
                                vid_text_mask = gr.Checkbox(
                                    label="Mask模式", value=False,
                                )
                        with gr.Column():
                            vid_text_output = gr.Video(label="跟踪结果")
                            vid_text_status = gr.Textbox(label="状态", interactive=False)

                    vid_text_btn.click(
                        track_video_text,
                        inputs=[vid_text_input, vid_text_prompt,
                                global_model, vid_text_mask, global_fa,
                                global_mmgp, global_mmgp_profile],
                        outputs=[vid_text_output, vid_text_status],
                        concurrency_limit=1,
                    )

                # ---------- 点击跟踪 ----------
                with gr.Tab("👆 点击跟踪"):
                    vid_pt_original = gr.State(None)
                    vid_pt_points = gr.State([])
                    vid_pt_frame_idx = gr.State(0)

                    with gr.Row():
                        with gr.Column():
                            vid_pt_video = gr.Video(
                                label="上传视频", sources=["upload"],
                            )
                            with gr.Row():
                                vid_pt_extract = gr.Button(
                                    "📸 提取首帧", variant="secondary",
                                )
                                vid_pt_browse = gr.Button(
                                    "🎞️ 选择帧", variant="secondary",
                                )
                            with gr.Group(visible=False) as vid_pt_browser:
                                vid_pt_preview = gr.Image(
                                    type="numpy", label="帧预览", interactive=False,
                                )
                                vid_pt_slider = gr.Slider(
                                    0, 1, value=0, step=1,
                                    label="帧位置", visible=False,
                                )
                                vid_pt_confirm = gr.Button(
                                    "✅ 确认选帧", variant="primary",
                                    visible=False,
                                )
                            vid_pt_frame = gr.Image(
                                type="numpy", label="标注帧（点击标记）",
                            )
                            vid_pt_type = gr.Radio(
                                ["正向点（前景）", "负向点（背景）"],
                                value="正向点（前景）", label="标记类型",
                            )
                            with gr.Row():
                                vid_pt_btn = gr.Button(
                                    "🎯 开始跟踪", variant="primary",
                                )
                                vid_pt_clear = gr.Button(
                                    "🗑️ 清除", variant="secondary",
                                )
                                vid_pt_mask = gr.Checkbox(
                                    label="Mask模式", value=False,
                                )
                        with gr.Column():
                            vid_pt_output = gr.Video(label="跟踪结果")
                            vid_pt_status = gr.Textbox(label="状态", interactive=False)

                    vid_pt_extract.click(
                        get_first_frame,
                        inputs=[vid_pt_video],
                        outputs=[vid_pt_frame, vid_pt_original, vid_pt_frame_idx, vid_pt_status],
                    )
                    vid_pt_browse.click(
                        get_video_total_frames,
                        inputs=[vid_pt_video],
                        outputs=[vid_pt_slider, vid_pt_preview,
                                 vid_pt_confirm, vid_pt_status],
                    ).then(
                        lambda: gr.update(visible=True),
                        outputs=[vid_pt_browser],
                    )
                    vid_pt_slider.release(
                        preview_frame,
                        inputs=[vid_pt_video, vid_pt_slider],
                        outputs=[vid_pt_preview, vid_pt_status],
                    )
                    vid_pt_confirm.click(
                        confirm_frame_selection,
                        inputs=[vid_pt_preview, vid_pt_slider],
                        outputs=[vid_pt_frame, vid_pt_original, vid_pt_frame_idx, vid_pt_status],
                    ).then(
                        lambda: gr.update(visible=False),
                        outputs=[vid_pt_browser],
                    )
                    vid_pt_frame.select(
                        on_video_frame_click,
                        inputs=[vid_pt_frame, vid_pt_original,
                                vid_pt_points, vid_pt_type],
                        outputs=[vid_pt_frame, vid_pt_points, vid_pt_status],
                    )
                    vid_pt_clear.click(
                        clear_video_points,
                        inputs=[vid_pt_original],
                        outputs=[vid_pt_frame, vid_pt_points, vid_pt_status],
                    )
                    vid_pt_btn.click(
                        track_video_points,
                        inputs=[vid_pt_video, vid_pt_points,
                                global_model, vid_pt_mask, global_fa,
                                global_mmgp, global_mmgp_profile,
                                vid_pt_frame_idx],
                        outputs=[vid_pt_output, vid_pt_status],
                        concurrency_limit=1,
                    )

                # ---------- 框选跟踪 ----------
                with gr.Tab("🔲 框选跟踪"):
                    gr.Markdown(
                        "在视频首帧点击画框：\n"
                        "- 🟢 **正向框**：圈选目标  |  🔴 **负向框**：排除区域\n"
                        "- 多框、含负向框或使用 SAM3.1 时自动走高层推理 API"
                    )
                    vid_box_original = gr.State(None)
                    vid_box_data = gr.State([])
                    vid_box_pending = gr.State(None)
                    vid_box_frame_idx = gr.State(0)

                    with gr.Row():
                        with gr.Column():
                            vid_box_video = gr.Video(
                                label="上传视频", sources=["upload"],
                            )
                            vid_box_text = gr.Textbox(
                                label="文本提示（可选）",
                                placeholder="例如: person, car — 留空则仅用框选",
                            )
                            vid_box_type = gr.Radio(
                                ["正向框（目标）", "负向框（排除）"],
                                value="正向框（目标）", label="框类型",
                            )
                            with gr.Row():
                                vid_box_extract = gr.Button(
                                    "📸 提取首帧", variant="secondary",
                                )
                                vid_box_browse = gr.Button(
                                    "🎞️ 选择帧", variant="secondary",
                                )
                            with gr.Group(visible=False) as vid_box_browser:
                                vid_box_preview = gr.Image(
                                    type="numpy", label="帧预览", interactive=False,
                                )
                                vid_box_slider = gr.Slider(
                                    0, 1, value=0, step=1,
                                    label="帧位置", visible=False,
                                )
                                vid_box_confirm = gr.Button(
                                    "✅ 确认选帧", variant="primary",
                                    visible=False,
                                )
                            vid_box_frame = gr.Image(
                                type="numpy", label="标注帧（点击画框）",
                            )
                            with gr.Row():
                                vid_box_btn = gr.Button(
                                    "🎯 开始跟踪", variant="primary",
                                )
                                vid_box_clear = gr.Button(
                                    "🗑️ 清除框", variant="secondary",
                                )
                                vid_box_mask = gr.Checkbox(
                                    label="Mask模式", value=False,
                                )
                        with gr.Column():
                            vid_box_output = gr.Video(label="跟踪结果")
                            vid_box_status = gr.Textbox(label="状态", interactive=False)

                    vid_box_extract.click(
                        get_first_frame,
                        inputs=[vid_box_video],
                        outputs=[vid_box_frame, vid_box_original, vid_box_frame_idx, vid_box_status],
                    )
                    vid_box_browse.click(
                        get_video_total_frames,
                        inputs=[vid_box_video],
                        outputs=[vid_box_slider, vid_box_preview,
                                 vid_box_confirm, vid_box_status],
                    ).then(
                        lambda: gr.update(visible=True),
                        outputs=[vid_box_browser],
                    )
                    vid_box_slider.release(
                        preview_frame,
                        inputs=[vid_box_video, vid_box_slider],
                        outputs=[vid_box_preview, vid_box_status],
                    )
                    vid_box_confirm.click(
                        confirm_frame_selection,
                        inputs=[vid_box_preview, vid_box_slider],
                        outputs=[vid_box_frame, vid_box_original, vid_box_frame_idx, vid_box_status],
                    ).then(
                        lambda: gr.update(visible=False),
                        outputs=[vid_box_browser],
                    )
                    vid_box_frame.select(
                        on_video_box_click,
                        inputs=[vid_box_frame, vid_box_original,
                                vid_box_data, vid_box_pending, vid_box_type],
                        outputs=[vid_box_frame, vid_box_data,
                                 vid_box_pending, vid_box_status],
                    )
                    vid_box_clear.click(
                        clear_video_boxes,
                        inputs=[vid_box_original],
                        outputs=[vid_box_frame, vid_box_data,
                                 vid_box_pending, vid_box_status],
                    )
                    vid_box_btn.click(
                        track_video_box,
                        inputs=[vid_box_video, vid_box_data,
                                global_model, vid_box_mask, vid_box_text, global_fa,
                                global_mmgp, global_mmgp_profile,
                                vid_box_frame_idx],
                        outputs=[vid_box_output, vid_box_status],
                        concurrency_limit=1,
                    )

        # 两个加载按鈕的事件绑定（必须定义在两个组件都创建完成之后）
        img_load_btn.click(
            load_image_model,
            inputs=[global_model, global_mmgp, global_mmgp_profile],
            outputs=[img_load_status, vid_load_status],
            concurrency_limit=1,
        )

        # SAM3.1 batch_size 联动：模型版本或 mmgp 开关变化时更新可见性和默认值
        def _update_batch_slider(model_version, use_mmgp):
            is_31 = model_version == "sam3.1"
            default_val = 4 if use_mmgp else 1
            return gr.update(visible=is_31), gr.update(visible=is_31, value=default_val)

        global_model.change(
            _update_batch_slider,
            inputs=[global_model, global_mmgp],
            outputs=[sam31_batch_md, sam31_batch_size],
        )
        global_model.change(
            lambda m: gr.update(visible=m != "sam3.1"),
            inputs=[global_model],
            outputs=[click_tab],
        )
        global_mmgp.change(
            _update_batch_slider,
            inputs=[global_model, global_mmgp],
            outputs=[sam31_batch_md, sam31_batch_size],
        )

        gr.Markdown(
            "---\n"
            "💡 **提示**: 所有功能均使用顶部选择的模型版本。"
            "首次使用某项功能会加载对应模型，请耐心等待。"
            "切换模型版本会卸载当前模型并重新加载。"
        )

    return demo


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    import socket

    def find_free_port(start=7860, end=7880):
        for port in range(start, end + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return port
        return start

    port = find_free_port()
    print(f"[SAM3] 使用端口: {port}")
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        inbrowser=True,
        ssr_mode=False,
    )

