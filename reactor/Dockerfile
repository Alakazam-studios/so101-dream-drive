# SO-101 Dream Drive on Reactor Runtime, built from source and a lockfile.
#
#   docker build -t so101-fd-reactor .
#   docker run --gpus all --network host so101-fd-reactor
#
# Then open http://<host>:8899/client.html and press Connect. One GPU with ~24 GB free; the
# model holds about 22 GB.
#
# Everything here comes from public sources: the framework from NVIDIA's repo at a pinned
# commit, the environment from that repo's own uv.lock, the SO-101 modules and the checkpoint
# from our HuggingFace repo. Nothing is mounted and nothing needs an account.
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

# ffmpeg is NOT optional: torchcodec dlopens libtorchcodec_core*.so against the system FFmpeg,
# and without it the loader fails with "Could not load this library" and never mentions FFmpeg.
#
# build-essential is NOT optional either, and this one is worse: the experiment compiles the
# tokenizer, so torch.compile invokes inductor at RUN time and shells out to a C compiler.
# Without it the container loads the model, reports session ready, accepts a WebRTC connection
# and then streams exactly one frame while every chunk dies with "InductorError: Failed to
# find C compiler".
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*
ENV CC=gcc CXX=g++

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

# The framework, pinned. A moving clone would silently change the environment the lock resolves.
ARG FW_SHA=5e67049cd94acb667786f1e6dd0dab821cb90c97
RUN git clone --filter=blob:none https://github.com/NVIDIA/cosmos-framework.git /opt/cosmos-framework \
    && git -C /opt/cosmos-framework checkout --quiet ${FW_SHA}

# THE ENVIRONMENT, FROM THE LOCK. The `cu130` group pins torch==2.10.0+cu130 and
# natten==0.21.6.dev6+cu130.torch210.gb300, both published wheels.
#
# The second command is the part that is not discoverable: the framework imports these five at
# inference time and the `serve` extra does not declare them. `iopath` is declared only under
# `train`, and it is reached through cosmos_framework.inference.args -> lazy_config.file_io, so
# a serving environment without it dies on import. Hitting these is what previously convinced
# us the environment was not resolvable at all.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /opt/cosmos-framework
#
# wandb is pinned at 0.28.1 rather than left free for a reason: reactor-runtime needs
# protobuf 7, and wandb below 0.28 pins protobuf 5, so the two stacks cannot share a process
# on an older wandb.
RUN uv sync --frozen --no-install-project --group cu130 --extra serve \
    && uv pip install --python /opt/venv/bin/python \
         iopath multi-storage-client boto3 qwen_vl_utils \
         "wandb==0.28.1" "reactor-runtime==3.1.2"

ENV HF_HOME=/opt/hfcache \
    PYTHONPATH=/app:/opt/cosmos-framework \
    PATH=/opt/venv/bin:/root/.local/bin:$PATH

# The SO-101 modules, and the checkpoint. `integration/` mirrors the tree layout, so it
# overlays directly onto the package.
RUN python -c "\
from huggingface_hub import snapshot_download;\
snapshot_download('alakazam-robonet/so101-fd-worldmodel',\
    allow_patterns=['integration/*','export_ft7600/*','patch_checkpoint_db.py'],\
    local_dir='/models')" \
    && cp -r /models/integration/* /opt/cosmos-framework/cosmos_framework/ \
    && rm -rf /models/integration

# Register the experiment. Shipped as a COPYed file rather than a heredoc: Cloud Build's
# legacy builder does not support `RUN cmd <<'EOF'`, and a heredoc whose first line is a
# python `from ... import ...` is parsed as a Dockerfile FROM instruction, which fails with
# "FROM requires either one or three arguments" pointing at a line inside the script.
COPY register_experiment.py /tmp/register_experiment.py
RUN python /tmp/register_experiment.py && rm /tmp/register_experiment.py

# Base model and VAE, both resolved BY NAME at load time rather than read from the checkpoint.
# Baked so serving never reaches for the network.
#
# Note the absence of `local_dir=` on the VAE. Passing it writes the file to that directory
# and NOT into the HuggingFace cache, so a by-name resolve still misses, falls through to the
# `uvx hf download` path, and fails offline with a CalledProcessError about Wan-AI rather than
# anything mentioning a cache. The file is fetched into the cache first, then copied out for
# WAN_VAE_PATH.
RUN python -c "\
import os, shutil;\
from huggingface_hub import snapshot_download, hf_hub_download;\
snapshot_download('nvidia/Cosmos3-Edge');\
p = hf_hub_download('Wan-AI/Wan2.2-TI2V-5B', 'Wan2.2_VAE.pth');\
os.makedirs('/opt/wan', exist_ok=True);\
shutil.copy(p, '/opt/wan/Wan2.2_VAE.pth');\
print('VAE cached and copied')"

# checkpoint_db._hf_download shells out to `uvx hf@1.16.4 download --format=json` with no
# local-cache short-circuit, inside a lifecycle hook. Offline that reads as a hang, not an
# error. This patches it to resolve from the cache first, and fails the build if the target
# signature moves.
RUN python /models/patch_checkpoint_db.py && rm /models/patch_checkpoint_db.py

COPY so101_dream.py so101_engine.py reactor.yaml /app/
COPY client.html /app/ui/client.html

ENV SO101_CKPT=/models/export_ft7600 \
    WAN_VAE_PATH=/opt/wan/Wan2.2_VAE.pth \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    NCCL_NET=Socket \
    REACTOR_LOG_LEVEL=INFO

# Fail the BUILD, not the first request. A GPU is not available here, so this checks
# everything that does not need one.
RUN python -c "\
import os, shutil, cosmos_framework, reactor_runtime, torch, natten;\
print('torch', torch.__version__, '| natten', natten.__version__);\
assert shutil.which('gcc'), 'no C compiler: torch.compile fails at run time, not build time';\
ck = os.environ['SO101_CKPT'];\
assert os.path.isdir(ck + '/anchors'), 'anchors missing';\
assert os.path.isfile(os.environ['WAN_VAE_PATH']), 'Wan VAE missing';\
import so101_dream;\
print('image OK')"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /app
EXPOSE 8080 8899
# The entrypoint puts torch's own CUDA libraries ahead of the system ones. Without it the
# container loads the model and reports session ready, then every chunk dies with
# CUBLAS_STATUS_NOT_INITIALIZED.
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash", "-c", "python -m http.server 8899 --directory /app/ui & exec python -m reactor_runtime.serve"]
