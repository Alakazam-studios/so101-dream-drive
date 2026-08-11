#!/usr/bin/env python3
"""SO-101 dream-driving session driver — v0.1.

Fork of runs/edge-causal/scripts/live_driver.py v1.3.2, stripped to ONE embodiment:
the campaign's own fine-tune served as a local HF export (vllm-omni --omni). What
changed vs the parent and why:

  * TRUE action control. The parent's umi mode modulated a vendor trajectory because
    hand-authored poses were off-manifold on stock Edge. The ft7600 fine-tune makes
    so101 absolute joint targets on-manifold (campaign gates 3/4), so sliders ARE the
    pose: 5 arm joints in LeRobot units plus a binary gripper. The server interpolates
    from the current pose toward the slider pose across the 16-step chunk with a
    per-step slew clamp, so user jerks cannot command off-corpus speeds.
  * Motion hand-off by default: the serve API accepts an mp4 input_reference; sending
    the last 5 generated frames in order is the campaign's measured-best hand-off
    (iter 33/34). Falls back to single-frame png until 5 frames exist.
  * Session discipline is inherited UNCHANGED from v1.2.x hotfixes: one live session,
    newest wins; reap-on-disconnect checked before every generate; start idempotent
    per socket; no silent closes.

Protocol (WS /session):
  client -> {"type":"start","anchor":"ep087","steps":int?,"guidance":float?,
             "handoff":"motion"|"single","seed":int?}
            {"type":"ctl","pose":[p1..p5],"grip":0|1}     latest-wins, ~10 Hz
            {"type":"reset","anchor":"ep024"}             re-condition on a real frame
            {"type":"stop"}
  server -> {"type":"ready","anchors":{...},"config":{...}}
            {"type":"chunk","n":i,"jpegs":[b64...],"w":..,"h":..,"gen_s":..,
             "rt":..,"pose":[...],"clamped":bool}
            {"type":"reset_ok","anchor":id} | {"type":"error","msg":..}
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi import WebSocket, WebSocketDisconnect
from PIL import Image

SERVE = os.environ.get("SO101_SERVE", "http://127.0.0.1:8000/v1/videos/sync")
WEIGHTS = Path(os.environ.get("SO101_WEIGHTS", "/weights/export_ft7600"))
ANCHOR_DIR = WEIGHTS / "anchors"
CLIENT_HTML = Path(__file__).parent / "client.html"

VERSION = "so101-dream v0.1"
CHUNK = 16
FPS = 15
SIZE = "640x480"
IMAGE_SIZE = 480
RAW_DIM = 6
DOMAIN = "so101"
VIEW = "third_person_view"
PROMPT = "pick up the blue object and place it in the yellow bin"
FLOW_SHIFT = 10.0
DEF_STEPS = 30
DEF_GUIDANCE = 1.0
JPEG_Q = 90
HANDOFF_FRAMES = 5              # motion hand-off: last 5 generated frames, in order

# Slew clamps, LeRobot units per 1/15 s step. Arm 6.0/step = 90 units/s, inside the
# corpus envelope; gripper 33/step closes fully in ~2 steps (binary toggle semantics).
ARM_SLEW = 6.0
GRIP_SLEW = 33.0
GRIP_OPEN, GRIP_CLOSED = 68.0, 2.0

app = FastAPI()
_ACTIVE: dict = {"task": None, "ws": None, "n": 0}


def _anchors() -> dict:
    meta = json.loads((ANCHOR_DIR / "anchors.json").read_text())
    return meta


def _sock_alive(ws) -> bool:
    try:
        from starlette.websockets import WebSocketState
        return (ws.client_state == WebSocketState.CONNECTED
                and ws.application_state == WebSocketState.CONNECTED)
    except Exception:
        return True


def _decode_mp4(b: bytes) -> np.ndarray:
    p = Path("/tmp/_live.mp4")
    p.write_bytes(b)
    import imageio.v3 as iio
    return np.asarray(iio.imread(p))


def _encode_ctx(frames: np.ndarray, path: Path, fps: int = FPS) -> Path:
    f = np.asarray(frames, dtype=np.uint8)
    t, h, w, _ = f.shape
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "0",
         "-preset", "ultrafast", "-pix_fmt", "yuv444p", str(path)],
        input=f.tobytes(), check=True)
    return path


def _jpegs(frames: np.ndarray, q: int = JPEG_Q) -> list[str]:
    out = []
    for fr in frames:
        b = io.BytesIO()
        Image.fromarray(np.asarray(fr, dtype=np.uint8)).save(b, "JPEG", quality=q)
        out.append(base64.b64encode(b.getvalue()).decode())
    return out


def build_chunk_actions(cur: np.ndarray, target: np.ndarray, n: int = CHUNK) -> tuple[list, np.ndarray, bool]:
    """n absolute action rows walking cur -> target under per-step slew clamps."""
    pose = cur.copy()
    rows, clamped = [], False
    for _ in range(n):
        d = target - pose
        lim = np.array([ARM_SLEW] * 5 + [GRIP_SLEW])
        step = np.clip(d, -lim, lim)
        if np.any(np.abs(d) > lim + 1e-9):
            clamped = True
        pose = pose + step
        rows.append(pose.tolist())
    return rows, pose, clamped


def gen_chunk(cond: Path | None, actions: list, seed: int, steps: int,
              guidance: float, cont: bool = False,
              size: str = SIZE, image_size: int = IMAGE_SIZE) -> tuple[np.ndarray, float, dict]:
    extra = {"action_mode": "forward_dynamics", "domain_name": DOMAIN,
             "action_chunk_size": len(actions), "image_size": image_size,
             "view_point": VIEW, "action": actions, "raw_action_dim": RAW_DIM,
             "guardrails": False}
    if cont:
        extra["continue"] = True                 # S1-deep: tensor residency
        extra["npy"] = True                      # capture already paid; take raw frames
    data = {"prompt": PROMPT, "num_frames": str(len(actions) + 1), "fps": str(FPS),
            "size": size, "num_inference_steps": str(steps),
            "guidance_scale": str(guidance), "flow_shift": str(FLOW_SHIFT),
            "seed": str(seed), "extra_params": json.dumps(extra)}
    t0 = time.perf_counter()
    if cont:
        r = requests.post(SERVE, data=data, headers={"Accept": "video/mp4"}, timeout=600)
    else:
        mime = "video/mp4" if str(cond).endswith(".mp4") else "image/png"
        with open(cond, "rb") as fh:
            r = requests.post(SERVE, data=data,
                              files={"input_reference": (Path(cond).name, fh, mime)},
                              headers={"Accept": "video/mp4"}, timeout=600)
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        raise RuntimeError(f"serve {r.status_code}: {r.text[:300]}")
    import json as _j
    phases = _j.loads(r.headers.get("X-Phase-Timings", "{}") or "{}")
    if r.headers.get("content-type", "").startswith("application/x-npy"):
        import io as _io
        frames = np.load(_io.BytesIO(r.content))             # (T,H,W,C) uint8
    else:
        frames = _decode_mp4(r.content)
    return frames, dt, phases


@app.get("/healthz")
def healthz():
    return {"ok": True, "version": VERSION, "domain": DOMAIN, "chunk": CHUNK,
            "fps": FPS, "size": SIZE, "steps_default": DEF_STEPS,
            "handoff_default": "motion", "single_session": True,
            "sessions_started": _ACTIVE["n"], "anchors": list(_anchors())}


@app.get("/")
def index():
    return HTMLResponse(CLIENT_HTML.read_text())


@app.get("/anchors/{name}")
def anchor_png(name: str):
    p = (ANCHOR_DIR / name).resolve()
    if not str(p).startswith(str(ANCHOR_DIR.resolve())) or not p.exists():
        return HTMLResponse("not found", status_code=404)
    return FileResponse(p)


@app.websocket("/session")
async def session(ws: WebSocket):
    await ws.accept()
    meta = _anchors()

    # v1.2.1 single-session policy: newest wins.
    old = _ACTIVE["task"]
    if old is not None and not old.done():
        old.cancel()
        if _ACTIVE["ws"] is not None:
            try:
                await _ACTIVE["ws"].send_text(json.dumps(
                    {"type": "error", "msg": "superseded by a newer session"}))
            except Exception:
                pass
    _ACTIVE["ws"] = ws
    _ACTIVE["n"] += 1

    state = {
        "ctl_pose": None,          # latest slider pose (6,) after grip expansion
        "running": False,
        "anchor": None,
        "steps": DEF_STEPS,
        "guidance": DEF_GUIDANCE,
        "handoff": "motion",
        "seed": 0,
        "chunk": CHUNK,
        "reset_to": None,
    }
    loop_task: asyncio.Task | None = None

    async def chunk_loop():
        try:
            anchor_id = state["anchor"]
            pose = np.asarray(meta[anchor_id]["pose"], dtype=np.float64)
            cond: Path = ANCHOR_DIR / f"{anchor_id}.png"
            cont = False
            n = 0
            t_start = time.time()
            while True:
                if not _sock_alive(ws):
                    return
                if time.time() - t_start > 600:            # public spend guardrail
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "msg": "session cap reached (10 min) — reconnect to keep driving"}))
                    return
                if state["reset_to"] is not None:
                    anchor_id = state["reset_to"]
                    state["reset_to"] = None
                    pose = np.asarray(meta[anchor_id]["pose"], dtype=np.float64)
                    cond = ANCHOR_DIR / f"{anchor_id}.png"
                    cont = False
                    await ws.send_text(json.dumps({"type": "reset_ok", "anchor": anchor_id}))
                target = pose if state["ctl_pose"] is None else np.asarray(state["ctl_pose"])
                actions, pose, clamped = build_chunk_actions(pose, target, int(state["chunk"]))
                frames, dt, phases = await asyncio.to_thread(
                    gen_chunk, cond, actions, state["seed"] + n, state["steps"],
                    state["guidance"], cont, state.get("size", SIZE),
                    state.get("image_size", IMAGE_SIZE))
                new = frames[1:]                       # frame 0 = conditioning duplicate
                # default "motion" = driver-side 5-frame clip (the receipt-holding
                # path: c12 x1.17, c16 x1.34). "resident" = engine tensor cache,
                # kept opt-in: it REGRESSED generate by ~0.25 s (receipts
                # s1deep_sweep.json) — cause unresolved, do not default.
                cont = state["handoff"] == "resident"
                if not cont:
                    if state["handoff"] == "motion":
                        buf = [f for f in new][-HANDOFF_FRAMES:]
                        if len(buf) >= min(HANDOFF_FRAMES, len(new)):
                            cond = _encode_ctx(np.stack(buf), Path("/tmp/_ctx.mp4"))
                    else:
                        Image.fromarray(new[-1]).save("/tmp/_last.png")
                        cond = Path("/tmp/_last.png")
                if not _sock_alive(ws):
                    return
                video_s = len(new) / FPS
                rt = video_s / dt
                # ADAPTIVE CHUNK (host speed varies ~±35% per boot, receipts
                # ship_480c12*.json): keep the stream realtime by sizing the chunk
                # to THIS host — grow when starving, shrink when there is headroom.
                if state.get("auto_chunk", True) and n >= 1:
                    hist = state.setdefault("rt_hist", [])
                    hist.append(rt)
                    if len(hist) >= 2:
                        avg = sum(hist[-3:]) / len(hist[-3:])
                        cur = int(state["chunk"])
                        if avg < 1.02 and cur < 32:
                            state["chunk"] = min(32, cur + 4)
                            hist.clear()
                        elif avg > 1.45 and cur > 12:
                            state["chunk"] = max(12, cur - 4)
                            hist.clear()
                await ws.send_text(json.dumps({
                    "type": "chunk", "n": n, "jpegs": _jpegs(new),
                    "w": int(new.shape[2]), "h": int(new.shape[1]),
                    "gen_s": round(dt, 3), "video_s": round(video_s, 3),
                    "rt": round(rt, 3), "chunk_len": int(state["chunk"]),
                    "pose": [round(float(x), 2) for x in pose],
                    "clamped": clamped, "phases": phases}))
                n += 1
        except asyncio.CancelledError:
            raise
        except Exception as e:                                    # no silent closes
            if _sock_alive(ws):
                try:
                    await ws.send_text(json.dumps({"type": "error", "msg": repr(e)[:400]}))
                except Exception:
                    pass

    try:
        try:
            gpu = requests.get(SERVE.replace("/v1/videos/sync", "/v1/models"), timeout=5).json().get("gpu", "?")
        except Exception:
            gpu = "?"
        await ws.send_text(json.dumps({
            "type": "ready", "version": VERSION, "gpu": gpu,
            "anchors": {k: {"episode": v["episode"], "pose": v["pose"]} for k, v in meta.items()},
            "config": {"chunk": CHUNK, "fps": FPS, "steps_default": DEF_STEPS,
                       "arm_slew": ARM_SLEW, "grip_open": GRIP_OPEN, "grip_closed": GRIP_CLOSED}}))
        while True:
            msg = json.loads(await ws.receive_text())
            t = msg.get("type")
            if t == "start":
                if state["running"]:                              # v1.2.2 idempotent start
                    continue
                state["anchor"] = msg.get("anchor", "ep087")
                state["steps"] = int(msg.get("steps", DEF_STEPS))
                state["guidance"] = float(msg.get("guidance", DEF_GUIDANCE))
                state["handoff"] = msg.get("handoff", "motion")
                state["seed"] = int(msg.get("seed", 0))
                state["chunk"] = max(4, min(64, int(msg.get("chunk_len", 16))))
                state["auto_chunk"] = bool(msg.get("auto_chunk", True))
                res = str(msg.get("res", "480"))
                state["size"], state["image_size"] = (("320x256", 256) if res == "256"
                                                      else ("640x480", 480))
                state["running"] = True
                loop_task = asyncio.create_task(chunk_loop())
                _ACTIVE["task"] = loop_task
            elif t == "ctl":
                arm = [float(x) for x in msg.get("pose", [])][:5]
                if len(arm) == 5:
                    grip = GRIP_CLOSED if int(msg.get("grip", 0)) else GRIP_OPEN
                    state["ctl_pose"] = arm + [grip]
            elif t == "reset":
                state["reset_to"] = msg.get("anchor", state["anchor"] or "ep087")
            elif t == "stop":
                break
    except WebSocketDisconnect:
        pass
    finally:
        if loop_task is not None and not loop_task.done():
            loop_task.cancel()
        if _ACTIVE["ws"] is ws:
            _ACTIVE["ws"] = None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8787")))
