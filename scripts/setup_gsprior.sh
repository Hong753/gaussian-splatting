#!/usr/bin/env bash
# Isolated plane rasterizer: never overwrite native 3DGS's CUDA extension.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gsprior_repo="${GSPRIOR_REPO:-$HOME/.cache/3dgs/GSPrior-dcb7c89-orange4f-v1}"
gsprior_env="${GSPRIOR_ENV:-$HOME/.venvs/gsprior-orange4f}"
# Reuse the working Git-installed extension from the active 3dgs environment.
# GSPrior only needs its distCUDA2 API, not its bundled simple-knn source tree.
gsprior_knn_extension="$(python - <<'PY'
from pathlib import Path
try:
    import torch
    import simple_knn._C as knn
except ImportError as error:
    raise SystemExit(f'Activate the working 3dgs environment with its existing simple_knn: {error}')
if not callable(getattr(knn, 'distCUDA2', None)):
    raise SystemExit('The installed simple_knn does not provide distCUDA2')
print(Path(knn.__file__).resolve())
PY
)"
if [[ ! -e "$gsprior_repo" ]]; then
  git clone https://github.com/takeshie/GSPrior.git "$gsprior_repo"
  git -C "$gsprior_repo" checkout --detach dcb7c89fb6b60f068b440de45d064ecc7fbcba55
fi
python scripts/setup_gsprior.py --repo "$gsprior_repo"
if [[ ! -x "$gsprior_env/bin/python" ]]; then python -m venv --system-site-packages "$gsprior_env"; fi
# Check even on a cached setup: a local package must not shadow the base install.
"$gsprior_env/bin/python" - "$gsprior_knn_extension" <<'PY'
import sys
from pathlib import Path
import torch
import simple_knn._C as knn
actual = Path(knn.__file__).resolve()
if actual != Path(sys.argv[1]).resolve():
    raise SystemExit(f'GSPrior simple_knn shadows the working 3dgs installation: {actual}; '
                     'use a fresh GSPRIOR_ENV to inherit the existing extension')
print(f'Reusing 3dgs simple_knn: {actual}', flush=True)
PY
if [[ ! -f "$gsprior_env/orange4f_ready" ]] || ! cmp -s "$gsprior_repo/orange4f_adapter.json" "$gsprior_env/orange4f_ready"; then
  "$gsprior_env/bin/python" -m pip install 'numpy<2' 'pycuda>=2024.1' numba scikit-image trimesh ninja plyfile pillow
  MAX_JOBS="${MAX_JOBS:-8}" "$gsprior_env/bin/python" -m pip install --no-build-isolation --no-deps \
    "$gsprior_repo/submodules/diff-plane-rasterization"
  "$gsprior_env/bin/python" -c 'import torch, pycuda.driver, diff_plane_rasterization, simple_knn._C; assert torch.cuda.is_available(); print(torch.__version__)'
  cp "$gsprior_repo/orange4f_adapter.json" "$gsprior_env/orange4f_ready"
fi
echo "GSPrior ready: $gsprior_env/bin/python; $gsprior_repo"
