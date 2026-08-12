#!/bin/bash
# Put torch's OWN CUDA libraries ahead of the system ones before anything imports torch.
#
# The base image ships a CUDA runtime, and the venv ships its own nvidia-* wheels. If the
# system copies win, torch loads a cuBLAS it was not built against and the first matmul dies
# with "CUDA error: CUBLAS_STATUS_NOT_INITIALIZED", which names neither the library nor the
# skew. The model still loads and the session still reports ready, so this only shows up when
# a frame has to be generated.
#
# Computed at start rather than hardcoded: the wheel layout under site-packages/nvidia moves
# between releases, and a stale hardcoded path fails the same silent way.
set -e

NVIDIA_LIBS="$(python - <<'PY'
import glob, os, site
paths = set()
for root in site.getsitepackages():
    for so in glob.glob(os.path.join(root, "nvidia", "**", "lib", "*.so*"), recursive=True):
        paths.add(os.path.dirname(so))
print(":".join(sorted(paths)))
PY
)"
TORCH_LIB="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"

export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec "$@"
