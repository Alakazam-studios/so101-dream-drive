#!/usr/bin/env python3
"""Persistent cosmos-framework inference engine behind the /v1/videos/sync contract.

Why this exists: the framework's cosmos3_omni export format is ahead of every public
vllm-omni release (three formats tried, receipts in the campaign ledger), so the
serving stack the campaign PROVED — cosmos_framework.scripts.inference — is wrapped
here with the model held resident. The endpoint speaks the same multipart contract
live_driver_so101 already uses, so the session layer is unchanged.

Runs inside the framework venv:  /cf/.venv/bin/python engine_server.py
Model load happens once at boot (the CLI reloads per call — that was the 40-60 s/chunk
tax on the drive demo). pipe.generate() is then called per request.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

CKPT = os.environ.get("SO101_CKPT", "/weights/export_ft7600")
PORT = int(os.environ.get("ENGINE_PORT", "8000"))
WORK = Path(tempfile.mkdtemp(prefix="engine_"))

# --- build the framework pipeline exactly the way the CLI does, once ---------------
sys.argv = ["inference", "-i", "/dev/null", "-o", str(WORK / "out"),
            "--checkpoint-path", CKPT, "--parallelism-preset", "latency",
            "--no-guardrails", "--seed", "0"]

from cosmos_framework.inference.common.init import init_script  # noqa: E402

init_script()

import tyro  # noqa: E402
from cosmos_framework.inference.common.args import tyro_cli  # noqa: E402
from cosmos_framework.inference.common.init import init_output_dir  # noqa: E402
from cosmos_framework.scripts.inference import InferenceArgs  # noqa: E402

_args = tyro_cli(InferenceArgs, description="", config=(tyro.conf.OmitArgPrefixes,))
_setup = _args.setup.build_setup()
init_output_dir(_setup.output_dir)
print(f"[engine] loading pipeline from {CKPT} ...", flush=True)
_t0 = time.time()
_pipe = _setup.get_inference_cls().create(_setup)
print(f"[engine] pipeline resident in {time.time() - _t0:.0f}s", flush=True)
_lock = threading.Lock()          # generation is serialized; one GPU, one stream
_n = 0

# --- S1-deep: PIXEL-TENSOR residency (no files, no ffmpeg, no contention) ----------
# The proven hand-off feeds 5 tail frames through read_media_frames -> pad-repeat to
# chunk+1. We cache those 5 frames as a uint8 (C,T,H,W) tensor at capture time (a
# tensor slice, ~micro-cost) and intercept read_media_frames on `continue` requests.
# Conditioning semantics are IDENTICAL to the receipt-holding mp4 path, minus the
# x264 encode + upload + decode.
import cosmos_framework.inference.action as _ACT  # noqa: E402
import cosmos_framework.inference.inference as _INF  # noqa: E402

_RES = {"cond": None, "full": None, "use": False, "want": False}
_orig_rmf = _ACT.read_media_frames


def _rmf(path, max_frames):
    if _RES["use"] and _RES["cond"] is not None:
        return _RES["cond"], 15.0
    return _orig_rmf(path, max_frames)


_ACT.read_media_frames = _rmf
_orig_save = _INF.save_img_or_video


def _save(vision_cthw, path, fps=None, quality=None, **kw):
    if not _RES["want"]:                                     # capture costs ~0.25 s
        return _orig_save(vision_cthw, path, fps=fps, quality=quality, **kw)
    try:
        import torch as _t
        t = vision_cthw.detach()
        if _t.is_floating_point(t):                          # save-time range is [0,1]
            t = (t.clamp(0, 1) * 255.0).round().to(_t.uint8)
        t = t.to("cpu")
        _RES["full"] = t                                     # (C,T,H,W) uint8
        _RES["cond"] = t[:, -5:].clone().contiguous()        # model input wants uint8
    except Exception as e:                                   # loud, never silent
        print(f"[residency] capture FAILED: {e!r}", flush=True)
    return _orig_save(vision_cthw, path, fps=fps, quality=quality, **kw)


_INF.save_img_or_video = _save
_STUB = WORK / "stub.png"


def generate_from_spec(spec: dict) -> tuple[bytes, float, dict]:
    """spec = the same dict the campaign's in.json files carried. Returns
    (mp4 bytes, generate seconds, per-phase timing dict)."""
    global _n
    with _lock:
        T = {}
        t = time.time()
        _n += 1
        name = f"req{_n:05d}"
        spec = dict(spec, name=name)
        sdir = _setup.output_dir / name
        sdir.mkdir(parents=True, exist_ok=True)
        spec_path = sdir / "in.json"
        spec_path.write_text(json.dumps(spec))
        overrides = _setup.get_sample_overrides_cls().from_files(
            [spec_path], overrides=_setup.sample_overrides)[0]
        overrides.output_dir = sdir
        T["build_overrides"] = round(time.time() - t, 3); t = time.time()
        overrides.download(sdir / "inputs")
        T["download"] = round(time.time() - t, 3); t = time.time()
        sample = overrides.build_sample(model_config=_pipe.model_config)
        T["build_sample"] = round(time.time() - t, 3); t = time.time()
        _pipe.generate([sample])
        dt = time.time() - t
        T["generate"] = round(dt, 3); t = time.time()
        out = sdir / name / "vision.mp4"
        if not out.exists():
            hits = list(sdir.rglob("vision.mp4"))
            if not hits:
                raise RuntimeError(f"no vision.mp4 under {sdir}")
            out = hits[0]
        data = out.read_bytes()
        T["read"] = round(time.time() - t, 3)
        shutil.rmtree(sdir, ignore_errors=True)
        return data, dt, T


def _frame_count(mp4: Path) -> int:
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames",
                        "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0", str(mp4)],
                       capture_output=True, text=True).stdout.strip()
    return int(r) if r.isdigit() else 0


# --- tiny HTTP layer matching /v1/videos/sync ---------------------------------------
from fastapi import FastAPI, Form, UploadFile, File, Response  # noqa: E402

api = FastAPI()


@api.get("/v1/models")
def models():
    try:
        import torch as _t
        gpu = _t.cuda.get_device_name(0)
    except Exception:
        gpu = "unknown"
    return {"data": [{"id": CKPT}], "resident_s": round(time.time() - _t0, 1), "gpu": gpu}


@api.post("/v1/videos/sync")
async def videos_sync(
    prompt: str = Form(...),
    num_frames: str = Form(...),
    fps: str = Form("15"),
    size: str = Form("640x480"),
    num_inference_steps: str = Form("30"),
    guidance_scale: str = Form("1.0"),
    flow_shift: str = Form("10.0"),
    seed: str = Form("0"),
    extra_params: str = Form("{}"),
    input_reference: UploadFile | None = File(None),
):
    extra = json.loads(extra_params)
    if extra.get("continue"):
        if _RES["cond"] is None:
            return Response(content="no resident context yet", status_code=409)
        _RES["use"] = True
        if not _STUB.exists():
            import PIL.Image as _PI
            _PI.new("RGB", (8, 8)).save(_STUB)
        ref = _STUB                                # never read; rmf intercepts
    elif input_reference is not None:
        ref = WORK / f"ref_{int(time.time()*1000)}_{input_reference.filename}"
        ref.write_bytes(await input_reference.read())
    else:
        return Response(content="no reference and no resident context",
                        status_code=409)
    spec = {
        "prompt": prompt,
        "fps": int(float(fps)),
        "image_size": int(extra.get("image_size", 480)),
        "model_mode": ("forward_dynamics" if extra.get("action_mode") == "forward_dynamics"
                       else extra.get("action_mode", "forward_dynamics")),
        "domain_name": extra.get("domain_name", "so101"),
        "view_point": extra.get("view_point", "third_person_view"),
        "action_chunk_size": int(extra.get("action_chunk_size", 16)),
        "raw_action_dim": int(extra.get("raw_action_dim", 6)),
        "seed": int(float(seed)),
        "vision_path": str(ref),
        "num_steps": int(float(num_inference_steps)),
        "guidance": float(guidance_scale),
    }
    if "action" in extra:
        apath = ref.with_suffix(".actions.json")
        apath.write_text(json.dumps(extra["action"]))
        spec["action_path"] = str(apath)
    _RES["want"] = bool(extra.get("continue") or extra.get("npy"))
    try:
        data, dt, T = generate_from_spec(spec)
    finally:
        _RES["use"] = False
        _RES["want"] = False
        if not extra.get("continue"):
            Path(ref).unlink(missing_ok=True)
    if extra.get("npy") and _RES["full"] is not None:
        import io as _io

        import numpy as _np
        thwc = _RES["full"].permute(1, 2, 3, 0).numpy()      # (T,H,W,C) uint8
        buf = _io.BytesIO()
        _np.save(buf, thwc)
        return Response(content=buf.getvalue(), media_type="application/x-npy",
                        headers={"X-Inference-Time-S": f"{dt:.3f}",
                                 "X-Phase-Timings": json.dumps(T)})
    return Response(content=data, media_type="video/mp4",
                    headers={"X-Inference-Time-S": f"{dt:.3f}",
                             "X-Phase-Timings": json.dumps(T)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="127.0.0.1", port=PORT, log_level="warning")
