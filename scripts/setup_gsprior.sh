#!/usr/bin/env bash
# Isolated plane rasterizer: never overwrite native 3DGS's CUDA extension.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gsprior_repo="${GSPRIOR_REPO:-$HOME/.cache/3dgs/GSPrior-dcb7c89-orange4f-v1}"
gsprior_env="${GSPRIOR_ENV:-$HOME/.venvs/gsprior-orange4f}"
if [[ ! -e "$gsprior_repo" ]]; then
  git clone https://github.com/takeshie/GSPrior.git "$gsprior_repo"
  git -C "$gsprior_repo" checkout --detach dcb7c89fb6b60f068b440de45d064ecc7fbcba55
fi
python scripts/setup_gsprior.py --repo "$gsprior_repo"
if [[ ! -x "$gsprior_env/bin/python" ]]; then python -m venv --system-site-packages "$gsprior_env"; fi
if [[ ! -f "$gsprior_env/orange4f_ready" ]] || ! cmp -s "$gsprior_repo/orange4f_adapter.json" "$gsprior_env/orange4f_ready"; then
  "$gsprior_env/bin/python" -m pip install 'numpy<2' 'pycuda>=2024.1' numba scikit-image trimesh ninja plyfile pillow
  MAX_JOBS="${MAX_JOBS:-8}" "$gsprior_env/bin/python" -m pip install --no-build-isolation --no-deps \
    "$gsprior_repo/submodules/diff-plane-rasterization" "$gsprior_repo/submodules/simple-knn"
  "$gsprior_env/bin/python" -c 'import torch, pycuda.driver, diff_plane_rasterization, simple_knn._C; assert torch.cuda.is_available(); print(torch.__version__)'
  cp "$gsprior_repo/orange4f_adapter.json" "$gsprior_env/orange4f_ready"
fi
echo "GSPrior ready: $gsprior_env/bin/python; $gsprior_repo"
