# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Ready-to-use inference wrappers around SAM3 / SAM3.1 (`sam3/` git submodule) for Windows + CUDA 12.8, with heavy emphasis on running 12GB-class GPUs via mmgp weight offloading. Code comments and user-facing strings are Chinese; keep that convention.

## Environment

The venv on disk is `SAM3mmgp_env/` (the README's install steps still say `SAM3easyuse_env` — stale). Always invoke `SAM3mmgp_env\python.exe`, never a bare `python`.

`*.bat` is gitignored, so the local launcher scripts are not in the repo — a fresh clone has none. They exist locally because the process needs PATH and env vars set before import; launching without them fails on DLL loads or HF downloads. The README documents a template under "封装成 API 的几个要点". Required:

- `PATH` must include `ffmpeg-8.1-full_build-shared\bin`, `SAM3mmgp_env\Library\bin`, `SAM3mmgp_env\Lib\site-packages\torch\lib`, `SAM3mmgp_env\Scripts`
- `HF_ENDPOINT=https://hf-mirror.com`, `HF_HOME` / `TORCH_HOME` pointed at `.huggingface`
- `XFORMERS_FORCE_DISABLE_TRITON=1`

Weights are not in the repo. `sam3/checkpoints/sam3.pt` and `sam3/checkpoints/sam3.1_multiplex.pt` must be downloaded separately; both are gitignored.

## Commands

Run these with the env vars above already set (that is what the local `.bat` wrappers do).

```bash
# Gradio Web UI on :7860 — note gradio_app.py binds 0.0.0.0
SAM3mmgp_env\python.exe gradio_app.py

# FastAPI grid-line service — --host defaults to 0.0.0.0 with no auth; pass 127.0.0.1 for local-only
SAM3mmgp_env\python.exe face_grid_api.py --host 127.0.0.1 --port 8000 --version 3.0

# CLI — subcommands: image-text, image-box, image-points, batch,
#                    video-text, video-points, video-box
SAM3mmgp_env\python.exe inference.py image-text -i photo.jpg -t "person, car" -o out.png
SAM3mmgp_env\python.exe inference.py video-box -v in.mp4 --box 100,50,400,300 --neg-box 10,10,80,80 -t "face" -o out.mp4
```

Shared CLI flags: `--model sam3|sam3.1`, `--mask` (binary mask instead of overlay), `--no-fa` (fall back to SDPA), `--mmgp`, `--mmgp-profile 1-5`, `--sam31-batch-size N`.

There is no test suite or linter for the wrapper code. The submodule carries its own: `cd sam3 && ..\SAM3mmgp_env\python.exe -m pytest test/test_io_utils.py`, with `black`/`ufmt`/`ruff` available via its `dev` extra. Verify wrapper changes by actually running the relevant CLI subcommand or UI tab against a real file.

## Architecture

Three independent entry points, each with its own copy of the model-lifecycle and mmgp logic. `gradio_app.py` and `face_grid_api.py` do **not** import `inference.py` — they duplicate it as module-level globals / service singletons. A fix to loading, unloading, or mmgp behavior usually needs to be mirrored in all three:

- [inference.py](inference.py) — `SAM3Inference` class + argparse CLI. The reference implementation.
- [gradio_app.py](gradio_app.py) — Web UI; same logic as module globals (`_image_processor`, `_video_predictors`, `_active_mode`, `_mmgp_applied`), keyed by version because the UI can switch models mid-session.
- [face_grid_api.py](face_grid_api.py) — FastAPI service overlaying a diagonal grid pattern on segmented faces/heads. Unlike the other two it has no mode switching: `__main__` loads **both** the video predictor and the image processor at startup and keeps them resident, applying mmgp inline. Endpoints are `async def` and push inference through `asyncio.to_thread`, where a `threading.Lock` (`_predictor_lock` / `_image_lock`) serializes it — the lock is for `inference_state` correctness, the thread hop keeps a 30s inference from blocking the event loop.

All three entry points prepend `sam3/` to `sys.path` at import time, so they run against the submodule source rather than only the installed package.

### Three mutually exclusive model modes

Only one of `image` / `interactive` / `video` is resident at a time. `_ensure_mode(mode)` unloads the other two, then `_cleanup_gpu()`. Every public method calls it first ([inference.py:411](inference.py#L411)). Models are lazy-loaded on first use via `_get_image_processor` / `_get_interactive` / `_get_video_predictor`.

Interactive point-click segmentation is SAM3-only — `sam3.1_multiplex.pt` lacks the weights, so `_get_interactive` hardcodes `checkpoint_sam3` regardless of the selected version ([inference.py:467](inference.py#L467)).

The video predictor is also rebuilt when `use_fa` changes, since Flash Attention is baked in at build time.

### mmgp offloading

`_apply_mmgp` wraps `mmgp.offload.profile()`, trying the module directly and then `{module_key: module}`. Returned offload objects are tracked in `_mmgp_applied` keyed by a dotted `target_name` (`image.sam3.1.model`, `video.sam3.tracker`) so `_release_mmgp_for(prefix)` can `release()` them on unload — without that, pinned host memory stays charged to "shared GPU" ([inference.py:377](inference.py#L377)).

Several non-obvious workarounds live here; they were fixed deliberately and are easy to break:

- Image models load with `pinnedMemory=True, budgets=None`. The default `budgets["transformer"]=1200` leaves submodules on CPU, and `predict_inst` bypasses the top-level `forward` to call `sam_prompt_encoder` directly, so mmgp's hooks never fire.
- Image models get a `Linear` forward pre-hook casting inputs to the weight dtype. `decoder.py`'s `forward_ffn` disables autocast, yielding float32 LayerNorm output against mmgp's BFloat16 weights ([inference.py:448](inference.py#L448)).
- Before offloading a video predictor, `_device` is force-cached to CUDA on the model and detector. `sam3_video_base.py`'s `device` property lazily caches `next(parameters()).device`; read after offload it returns CPU and images get moved off-GPU ([inference.py:582](inference.py#L582)).
- For SAM3.1 video, `offload_output_to_cpu_for_eval` and `trim_past_non_cond_mem_for_eval` are enabled on the inner tracker, and `use_batched_grounding` / `batched_grounding_batch_size` / `postprocess_batch_size` follow `sam31_batch_size`. CLI default is 4 (upstream default is 16); use 1 with mmgp for the lowest VRAM.

`_cleanup_gpu` prefers `mmgp.offload.flush_torch_caches()`, falling back to `torch._C._host_emptyCache()` plus a `ctypes` `EmptyWorkingSet()` call — on Windows nothing else makes the shared-GPU number actually drop ([inference.py:337](inference.py#L337)).

### Two video tracking APIs

Video paths choose between a high-level and a low-level API, and this choice drives most of the branching in `track_video_*`:

- **High-level** `predictor.handle_request` / `handle_stream_request` (start_session → add_prompt → propagate_in_video → close_session). Required for SAM3.1, any text prompt, multiple boxes, or negative boxes.
- **Low-level** `tracker.add_new_points_or_box` + `tracker.propagate_in_video`, via `_get_tracker()`, which grafts `model.detector.backbone` onto the tracker. Used only for SAM3 + no text + a single positive box.

The initial visual prompt accepts exactly one box (`sam3_video_inference.py::_get_visual_prompt`). Multi-box requests therefore add the first box, read `out_obj_ids[0]`, then add each remaining box as a refinement on that same `obj_id` with `clear_old_boxes=False` ([inference.py:1262](inference.py#L1262)). Boxes go to the high-level API as normalized **xywh** with `bounding_box_labels` 1/0 for positive/negative; the low-level API takes normalized **xyxy**.

Sessions are closed in a `finally` block — leaking one holds GPU state for the process lifetime.

All inference runs under `torch.autocast("cuda", dtype=torch.bfloat16)`.

### Video I/O

`_read_video_frames` loads every frame into RAM as RGB, so memory scales with video length. Propagation results are collected per frame, rendered, then written by `_write_video`: OpenCV `mp4v` to a `.tmp.mp4`, then ffmpeg transcode to H.264 `yuv420p` for browser playability, falling back to the raw temp file if ffmpeg is missing. `ffmpeg` is invoked by name and resolved from PATH.
