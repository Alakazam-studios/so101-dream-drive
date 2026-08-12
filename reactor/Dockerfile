# SO-101 Dream Drive, packaged for Reactor Runtime: one image, nothing to mount.
#
#   docker run --gpus all -p 8080:8080 -p 8899:8899 -p 10000-60000:10000-60000/udp \
#     us-central1-docker.pkg.dev/ewilan-pipeline/so101/so101-fd-reactor:1.0
#
# Then open http://<host>:8899/client.html and press Connect. The weights are baked in, so
# there is no download step, no HuggingFace account and no volume to wire up.
#
# The base image is the environment we train and serve with: torch 2.10.0+cu130 with a custom
# NATTEN build, the cosmos-framework tree with the SO-101 modules registered, FFmpeg, and a
# prepopulated HuggingFace cache holding the Cosmos3-Edge base and the Wan 2.2 VAE. It is
# published for anonymous pull. Building that environment from a requirements file does not
# reproduce it, which is why it ships as bytes.
FROM us-central1-docker.pkg.dev/ewilan-pipeline/so101/so101-fd:1.0

# A C compiler, which is NOT optional and is not obvious. The experiment compiles the
# tokenizer (`compile_tokenizer=dict(enabled=True, warmup_resolutions=["480"])`), so
# torch.compile invokes inductor at RUN time, and inductor shells out to a C compiler. The
# base image has no toolchain, so without this every chunk dies with
# "InductorError: Failed to find C compiler" while the container still loads the model,
# reports `session ready`, accepts a WebRTC connection and streams exactly one keyframe.
# Nothing about that failure points at a missing compiler.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
ENV CC=gcc CXX=g++

# The runtime, plus the one dependency that decides whether these two stacks can share a
# process at all: reactor-runtime needs protobuf 7, and wandb below 0.28 pins protobuf 5, so
# importing both in one interpreter fails on the older wandb. These are the exact versions
# verified serving this checkpoint end to end.
RUN python3.13 -m pip install --no-cache-dir --upgrade --target=/opt/sitepkgs \
        "reactor-runtime==3.1.2" "wandb==0.28.1"

COPY so101_dream.py so101_engine.py reactor.yaml /app/
COPY client.html /app/ui/client.html

# The checkpoint, from the public model repo. Two things matter here:
#
#   * HF_HUB_OFFLINE is 1 in the base image, because serving must never reach for the network.
#     It is relaxed for this layer only.
#   * The weights go to /models, NOT /weights. The base image declares VOLUME ["/weights"], and
#     Docker discards everything a later layer writes into a declared volume: the download
#     reports success, the layer commits, and the files are simply gone by the next RUN. That
#     failure mode names nothing and looks like a broken download. /weights stays free as a
#     mount point for anyone who wants to override the baked checkpoint.
RUN HF_HUB_OFFLINE=0 python3.13 -c "\
from huggingface_hub import snapshot_download;\
snapshot_download('alakazam-robonet/so101-fd-worldmodel', allow_patterns=['export_ft7600/*'],\
                  local_dir='/models')"

ENV SO101_CKPT=/models/export_ft7600 \
    NCCL_NET=Socket \
    REACTOR_LOG_LEVEL=INFO \
    PYTHONPATH=/app:/opt/cosmos-framework:/opt/sitepkgs

# Fail the BUILD, not the first request, if the two stacks cannot coexist or the weights did
# not land. This is the check that catches a protobuf regression on a dependency bump.
RUN python3.13 -c "\
import os, shutil, cosmos_framework, reactor_runtime, google.protobuf as pb;\
print('cosmos + reactor import OK, protobuf', pb.__version__);\
assert shutil.which('gcc'), 'no C compiler: torch.compile fails at run time, not build time';\
ck = os.environ['SO101_CKPT'];\
assert os.path.isdir(ck + '/anchors'), 'anchors missing under ' + ck;\
assert os.path.isfile(ck + '/config.json'), 'export config.json missing under ' + ck;\
assert os.path.isfile('/app/reactor.yaml'), 'app missing';\
import so101_dream;\
print('image OK, pipeline imports')"

WORKDIR /app
EXPOSE 8080 8899

# The runtime serves signalling and media on 8080; the little static server hands out the
# page on 8899 so a browser has something to load without a second container.
CMD ["bash", "-c", "python3.13 -m http.server 8899 --directory /app/ui & exec python3.13 -m reactor_runtime.serve"]
