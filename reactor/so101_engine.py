"""The cosmos-framework side of the Reactor port: load once, generate a chunk, hand back
frames as numpy arrays.

Kept separate from the pipeline so the Reactor-facing file stays about the robot and the
stream, and everything framework-specific (the load-once dance, the spec dict, the mp4
round trip) lives here.

Two details that are not guessable and cost us a day each when we got them wrong:

  * ``init_script()`` must run BEFORE the setup is built. It is what registers the
    experiment configs; skip it and the checkpoint resolves to a MissingConfigException
    that never mentions SO-101.
  * the framework writes an mp4 and returns a path, so a chunk round-trips through
    libx264 whether you want it or not. We decode it straight back to RGB. It is wasteful
    and it is also the supported path; the alternative is reaching into the sampler.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

DEFAULT_CKPT = "/weights/export_ft7600"


class Engine:
    """A resident cosmos-framework pipeline."""

    def __init__(self, config_path: Path | None = None) -> None:
        ckpt = DEFAULT_CKPT
        if config_path and Path(config_path).exists():
            cfg = json.loads(Path(config_path).read_text())
            ckpt = cfg.get("checkpoint_path", ckpt)
        self.ckpt = ckpt

        # init_script FIRST — it registers the experiment configs the checkpoint needs.
        from cosmos_framework.inference.common.init import init_script

        init_script()
        from cosmos_framework.inference.args import OmniSetupOverrides

        setup = OmniSetupOverrides.model_construct()
        setup.checkpoint_path = ckpt
        setup.output_dir = tempfile.mkdtemp(prefix="so101_")
        setup.parallelism_preset = "latency"
        setup.guardrails = False          # OpenMDW-1.1 carries no guardrail obligation
        t0 = time.time()
        self.setup = setup.build_setup()
        self.pipe = self.setup.get_inference_cls().create(self.setup)
        self.load_s = round(time.time() - t0, 1)
        self._n = 0

    def default_anchor(self) -> tuple[np.ndarray, np.ndarray]:
        """The starting frame and the pose that produced it.

        Anchors ship beside the weights (``export_ft7600/anchors/``): a real photograph
        plus its joint values, so a session starts from something the model has actually
        seen rather than from noise.
        """
        adir = Path(self.ckpt) / "anchors"
        meta = json.loads((adir / "anchors.json").read_text())
        key = sorted(meta)[0]
        png = next(adir.glob(f"{key}*.png"), None) or next(adir.glob("*.png"))
        frame = _decode_image(png)
        pose = np.array(meta[key]["pose"], dtype=np.float32)
        return frame[None, ...], pose

    def generate(self, cond, actions, prompt, domain, view, steps, guidance, fps) -> list[np.ndarray]:
        """One chunk: conditioning frames plus absolute action rows in, RGB frames out."""
        self._n += 1
        work = Path(self.setup.output_dir) / f"req{self._n:05d}"
        work.mkdir(parents=True, exist_ok=True)
        try:
            cond_path = _encode(np.asarray(cond), work / "cond.mp4", fps)
            spec = {
                "name": f"req{self._n:05d}",
                "prompt": prompt,
                "action_mode": "forward_dynamics",
                "domain_name": domain,
                "view_point": view,
                "action": actions,
                "action_chunk_size": len(actions),
                "raw_action_dim": len(actions[0]),
                "num_frames": str(len(actions) + 1),
                "fps": str(fps),
                "num_steps": steps,
                "guidance": guidance,
                "seed": 0,
                "vision_path": str(cond_path),
                "output_dir": str(work),
            }
            spec_path = work / "in.json"
            spec_path.write_text(json.dumps(spec))
            overrides = self.setup.get_sample_overrides_cls().from_files(
                [spec_path], overrides=self.setup.sample_overrides)[0]
            overrides.output_dir = work
            overrides.download(work / "inputs")
            sample = overrides.build_sample(model_config=self.pipe.model_config)
            self.pipe.generate([sample])
            out = next(work.rglob("vision.mp4"), None)
            if out is None:
                raise RuntimeError(f"no vision.mp4 produced under {work}")
            return _decode(out)
        finally:
            shutil.rmtree(work, ignore_errors=True)


def _encode(frames: np.ndarray, path: Path, fps: int) -> Path:
    """RGB frames -> mp4, the only conditioning format the sample builder accepts."""
    h, w = frames.shape[1:3]
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", str(path)],
        stdin=subprocess.PIPE)
    p.communicate(np.ascontiguousarray(frames, dtype=np.uint8).tobytes())
    if p.returncode:
        raise RuntimeError("conditioning encode failed")
    return path


def _decode(path: Path) -> list[np.ndarray]:
    """mp4 -> list of RGB frames."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(x) for x in probe.split(",")[:2])
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    return list(np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3))


def _decode_image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
