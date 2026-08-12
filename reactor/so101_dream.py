"""SO-101 Dream Drive on the Reactor Runtime.

A world model as a live stream: the sliders are absolute joint targets, the model
generates the frames that follow from them, and the arm you are driving does not exist.

This is a port of our own WebSocket driver. Everything that was transport there
(session lifecycle, frame pacing, command validation, the HTML page) is deleted, because
the runtime supplies it. What survives is the part that is actually about the robot:

  * the slew clamp. Sliders are ABSOLUTE targets, so a client can jerk one from -100 to
    +100 in a single message. The model was trained on teleop at human speeds, and a
    step that large is off-corpus, so the chunk WALKS from the current pose toward the
    target under a per-step limit instead of jumping. 6.0 units/step is 90 units/s, inside
    the corpus envelope; the gripper gets 33/step so it closes in ~2 steps, which is the
    binary open/closed semantics the training data has.
  * the motion hand-off. Each chunk is conditioned on the tail of the previous one, so
    the arm continues from where it visibly is rather than snapping to an anchor.
  * generating a chunk ahead. One chunk is 16 frames at 15 fps, about 1.07 s of motion,
    and generating it takes longer than playing it. A worker thread produces chunk N+1
    while the stream plays chunk N; without it the stream stalls between every chunk.

The model is a forward-dynamics world model, not a controller: it answers "what happens
if I command this", so a client supplies the intent and watches the consequence.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from reactor_runtime import Idle, InputField, InputState, Output, ReactorPipeline, Video, event
from reactor_runtime.interface.pipeline.idle import _IdleType
from reactor_runtime.log import get_logger

logger = get_logger(__name__)

FPS = 15
CHUNK = 16                      # actions per generated chunk (~1.07 s of motion)
DOMAIN = "so101"
VIEW = "third_person_view"
PROMPT = "pick up the blue object and place it in the yellow bin"
RAW_DIM = 6
DEF_STEPS = 30
DEF_GUIDANCE = 1.0
# Tail of the previous chunk handed to the next as conditioning. MEASURED, not reasoned:
# a hold-pose A/B (same anchor, same actions, same seed, 8 chunks) puts positional drift at
# parity (MSE vs anchor 273 at h=1 against 274 at h=5) but separates sharply on texture —
# Laplacian variance falls 751 -> 681 at h=1 and holds at ~740 from chunk 2 onward at h=5,
# i.e. 15.8% of the anchor's high-frequency energy lost versus 8.5%. The single-frame version
# melts. Reading the framework's `condition_frame_indexes = list(range(n))` suggests only the
# leading frame should matter, so this contradicts the code; trust the measurement.
HANDOFF_FRAMES = 5

# LeRobot units per 1/15 s step, measured against the training corpus envelope.
ARM_SLEW = 6.0
GRIP_SLEW = 33.0
GRIP_OPEN, GRIP_CLOSED = 68.0, 2.0

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


class DreamOutput(Output):
    """The generated view of the arm."""

    main_video: Video


class SO101State(InputState):
    """What a client can change mid-stream.

    Each field becomes a ``set_<field>`` command, so the whole control surface is
    declared here rather than in a hand-written protocol. Ranges are the LeRobot
    normalized units the model was trained on; values outside them are off-corpus and the
    runtime rejects them before the handler runs.
    """

    shoulder_pan: float = InputField(default=0.0, ge=-100.0, le=100.0, description="Base rotation.")
    shoulder_lift: float = InputField(default=0.0, ge=-100.0, le=100.0, description="Shoulder pitch.")
    elbow_flex: float = InputField(default=0.0, ge=-100.0, le=100.0, description="Elbow.")
    wrist_flex: float = InputField(default=0.0, ge=-100.0, le=100.0, description="Wrist pitch.")
    wrist_roll: float = InputField(default=0.0, ge=-100.0, le=100.0, description="Wrist rotation.")
    grip: str = InputField(default="open", choices=["open", "closed"], description="Gripper.")
    paused: bool = InputField(default=False, description="Stop generating; hold the last frame.")


class SO101DreamDrive(ReactorPipeline):
    """Stream a world model of an SO-101 arm, driven live by joint targets."""

    state: SO101State
    fps = FPS

    def load(self, config_path: Path | None) -> None:
        """Build the inference pipeline once. This is the expensive step (~12 s on an
        H100), which is why the runtime's warm session model matters for this workload:
        paid once per container, not once per viewer."""
        from so101_engine import Engine        # thin wrapper over cosmos-framework

        self.engine = Engine(config_path)
        self.pose = np.zeros(RAW_DIM, dtype=np.float32)
        self.pose[5] = GRIP_OPEN
        self.anchor = self.engine.default_anchor()   # (frames, pose) the reset returns to
        self.cond = self.anchor[0]
        self.pose = self.anchor[1].astype(np.float32).copy()
        self._frames: queue.Queue = queue.Queue(maxsize=CHUNK * 3)
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        # Bumped by every reset. A chunk that was already in flight when the reset landed
        # carries the old epoch and is thrown away instead of overwriting the fresh anchor —
        # otherwise a reset issued mid-chunk is silently undone a second later by the
        # generation it interrupted, which makes the button look like it does nothing.
        self._epoch = 0
        # The worker is NOT started here: `state` is bound after load() returns, and a
        # thread that reads it immediately dies on NoneType. It starts on the first
        # inference() turn, when the state is guaranteed to exist.
        logger.info("so101 dream drive ready", fps=FPS, chunk=CHUNK)

    @event(name="reset_to_anchor", description="Return to the starting frame and pose.")
    def reset_to_anchor(self) -> None:
        """Drop the generated history and start again from a real photograph.

        Long rollouts drift; this is the escape hatch that costs nothing.
        """
        with self._lock():
            self.cond = self.anchor[0]
            self.pose = self.anchor[1].astype(np.float32).copy()
            self._epoch += 1
            self._drain()
        logger.info("reset to anchor", epoch=self._epoch)

    def _lock(self):
        if not hasattr(self, "_mutex"):
            self._mutex = threading.Lock()
        return self._mutex

    def _drain(self) -> None:
        while not self._frames.empty():
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break

    def _target(self, s: SO101State) -> np.ndarray:
        """The slider positions as one absolute action row."""
        grip = GRIP_CLOSED if s.grip == "closed" else GRIP_OPEN
        return np.array(
            [s.shoulder_pan, s.shoulder_lift, s.elbow_flex, s.wrist_flex, s.wrist_roll, grip],
            dtype=np.float32,
        )

    def _walk(self, target: np.ndarray) -> tuple[list, np.ndarray]:
        """CHUNK absolute rows walking the current pose toward *target* under the slew
        clamp. A client jerking a slider gets a fast but on-manifold move, not a
        teleport the model has never seen."""
        pose = self.pose.copy()
        lim = np.array([ARM_SLEW] * 5 + [GRIP_SLEW], dtype=np.float32)
        rows = []
        for _ in range(CHUNK):
            pose = pose + np.clip(target - pose, -lim, lim)
            rows.append(pose.tolist())
        return rows, pose

    def _produce(self) -> None:
        """Generate chunk N+1 while the stream is still playing chunk N.

        Every read of ``self.state`` happens INSIDE the try. The runtime clears the state to
        None between sessions, so a read out here raises AttributeError on the way to
        ``Thread._bootstrap_inner`` and kills the producer for the life of the process — after
        which the container still loads, still accepts sessions, and still answers SDP, but
        serves exactly one keyframe to every client that ever connects again. It cost us an
        afternoon precisely because nothing about the symptom points at a dead thread.
        """
        while not self._stop.is_set():
            try:
                state = self.state
                if state is None or state.paused or self._frames.qsize() > CHUNK:
                    self._stop.wait(0.05)
                    continue
                with self._lock():
                    cond, start, epoch = self.cond, self.pose.copy(), self._epoch
                rows, end = self._walk(self._target(state))
                frames = self.engine.generate(
                    cond=cond, actions=rows, prompt=PROMPT, domain=DOMAIN, view=VIEW,
                    steps=DEF_STEPS, guidance=DEF_GUIDANCE, fps=FPS,
                )
                with self._lock():
                    if epoch != self._epoch:
                        continue                             # reset landed mid-chunk; drop it
                    self.pose = end
                    self.cond = frames[-HANDOFF_FRAMES:]     # motion hand-off
                for f in frames:
                    self._frames.put(f)
            except Exception:                                 # a bad chunk must not end the stream
                logger.exception("chunk generation failed; holding the last frame")
                self._stop.wait(0.5)

    def inference(self) -> Iterator[DreamOutput | _IdleType]:
        """Hand the runtime one frame per turn; Idle while paused or between chunks.

        Idle rather than a blocking wait: the runtime keeps the session alive and paces
        the stream, so a slow chunk shows as a hold rather than a stall or a dropped
        connection.
        """
        # RESEED. inference() is entered once per session, so this is "on connect": drop the
        # conditioning and pose back to a real photograph. Without it a session inherits
        # whatever the previous viewer's rollout drifted into, and the stream opens mangled on
        # a fresh connection while every log on both sides looks healthy.
        self.reset_to_anchor()

        # Restart the producer if it is missing OR dead.
        if self._worker is None or not self._worker.is_alive():
            self._stop.clear()
            self._worker = threading.Thread(target=self._produce, daemon=True)
            self._worker.start()
            logger.info("producer started")
        last: np.ndarray | None = None
        while True:
            state = self.state
            if state is None or state.paused:
                yield Idle
                continue
            try:
                last = self._frames.get(timeout=0.02)
            except queue.Empty:
                yield Idle
                continue
            yield DreamOutput(main_video=last)


if __name__ == "__main__":
    import json

    from reactor_runtime.interface.model import ModelContract

    print(json.dumps(ModelContract.of(SO101DreamDrive).render_schema().to_openapi(), indent=2))
