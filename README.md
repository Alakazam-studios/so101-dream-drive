# SO-101 Dream Drive

Drive a robot arm that does not exist. You move the sliders, a world model generates the
next frames, and the arm in the video responds. Every frame is generated. Nothing is
rendered, and there is no simulator underneath.

The model is [alakazam-robonet/so101-fd-worldmodel](https://huggingface.co/alakazam-robonet/so101-fd-worldmodel),
a Cosmos 3 Edge fine-tune on 254 episodes of a 3D-printed SO-101 doing pick-and-place.
Results and limitations: [alakazam.gg/so-wam](https://alakazam.gg/so-wam).

## What this is for

It is a working example of the workload, not a product: a session opened over a WebSocket,
held warm, driven interactively at ~1 s per chunk. If you are evaluating whether to host
this kind of model, this is the smallest thing that shows the shape of it, including the
part that matters: a cold start costs minutes and a warm session costs seconds.

## Pieces

| file | what it does |
|---|---|
| `engine_server.py` | Loads the model once, serves `POST /v1/videos/sync`. The expensive part. |
| `live_driver_so101.py` | WebSocket `/session`: takes slider positions, builds action chunks, calls the engine, streams frames back. Also serves the page. |
| `client.html` | The page. Sliders, connect, grip toggle, reset-to-anchor, stop. 166 lines, no build step. |

Two processes on purpose: the engine holds a 4B model and takes ~12 s to load, so it stays
resident while drivers come and go.

## Running it

You need one GPU with ~24 GB free, Docker, and the weights.

**1. Get the runtime.** Public, no account needed:

```bash
docker pull us-central1-docker.pkg.dev/ewilan-pipeline/so101/so101-fd:1.0
```

It carries torch 2.10.0+cu130 with a custom NATTEN build, the cosmos-framework tree with
the SO-101 modules registered, FFmpeg, and a prepopulated HuggingFace cache. Building this
environment from a requirements file does not work; we shipped the bytes for a reason.

**2. Get the weights.** This demo wants the **export** format (a directory with its own
`config.json`), not the raw DCP checkpoint:

```bash
huggingface-cli download alakazam-robonet/so101-fd-worldmodel \
  --include "export_ft7600/*" --local-dir /data/so101
```

`export_ft7600/anchors/` holds the anchor frames that "reset to anchor" returns to.

**3. Start the engine**, then the driver:

```bash
docker run --gpus all -d --name so101-engine --network host \
  -v /data/so101/export_ft7600:/weights/export_ft7600:ro \
  us-central1-docker.pkg.dev/ewilan-pipeline/so101/so101-fd:1.0 \
  python3.13 /app/engine_server.py

docker run --gpus all -d --name so101-driver --network host \
  -v /data/so101/export_ft7600:/weights/export_ft7600:ro \
  -v "$PWD":/app -w /app \
  us-central1-docker.pkg.dev/ewilan-pipeline/so101/so101-fd:1.0 \
  python3.13 live_driver_so101.py
```

Open `http://localhost:8787` and press Connect.

Environment variables, if your paths differ:

| var | default |
|---|---|
| `SO101_CKPT` | `/weights/export_ft7600` (engine) |
| `SO101_WEIGHTS` | `/weights/export_ft7600` (driver, and where it finds `anchors/`) |
| `SO101_SERVE` | `http://127.0.0.1:8000/v1/videos/sync` |
| `PORT` | `8787` |

## What you should see

Move a slider and the arm moves that joint. Toggle the grip and the gripper closes. Reset
returns to an anchor frame. Frames arrive in chunks of 16 at 15 fps, so roughly one second
of motion per round trip.

First request after a cold start takes minutes, because the model load dominates. After
that, each chunk is seconds. That gap is the whole argument for session-oriented serving.

## What it does badly

Worth knowing before you judge the model by this demo.

**It was trained on one fixed camera in one room.** Move the viewpoint, change the table or
the lighting, and it degrades sharply. Not gracefully, either: a sibling policy in the
same project stopped moving entirely when a prop changed colour from red to yellow.

**It is a world model, not a controller.** It predicts what happens if you command
something. It does not decide what to command.

**Long horizons drift.** It is trained and evaluated on ~1 s chunks. Chain many and quality
falls off; the end effector is its worst-rendered region by roughly 2.2× versus the frame
as a whole.

## Numbers

Held-out test: anchor frame plus one second of real commanded joint angles, measure where
the end effector lands versus the real arm.

| | end-effector error |
|---|---|
| this fine-tune | 5.8 px |
| a LoRA fine-tune, same base and data | 14.0 px |
| stock Cosmos 3 Edge | 132.3 px |

25 held-out episodes, wins all 25. The middle row is the honest comparison: most of the gap
is buying the fine-tune at all, not this particular recipe.

## License

Model weights inherit **OpenMDW-1.1** from `nvidia/Cosmos3-Edge`, which carries no
guardrail obligation and no attribution requirement on outputs. Note that other Cosmos SKUs (Transfer, Predict)
use NVIDIA's Open Model License, which has a self-executing termination clause; terms do
not transfer between them. Code in this repo is MIT.
