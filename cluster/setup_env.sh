#!/bin/bash
# One-time environment setup. Run on a LOGIN node (needs internet):
#   bash cluster/setup_env.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/cluster/env.sh"

echo ">> Fetching the LIBERO submodule"
git -C "$REPO_DIR" submodule update --init --recursive

echo ">> Creating virtualenv at $VENV_DIR"
virtualenv --no-download "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --no-index --upgrade pip

echo ">> Checking the module-provided packages (python $(python -c 'import sys;print(sys.version.split()[0])'))"
python -c "import mujoco, cv2; print('mujoco', mujoco.__version__, '| opencv', cv2.__version__)" || {
    echo "ERROR: mujoco/opencv not importable. Run 'module spider mujoco' to find a version" >&2
    echo "       that supports this Python, and set it in cluster/env.sh." >&2
    exit 1
}

echo ">> Installing BadVLA / OpenVLA-OFT dependencies"
# flash-attn is intentionally skipped: evaluation doesn't use it.
pip install -e "$REPO_DIR"
# dlimp pins tensorflow==2.15.0, which Alliance doesn't provide (only 2.15.1).
# Its only deps (tensorflow, tensorflow_datasets) are already installed above.
pip install --no-deps "dlimp @ git+https://github.com/moojink/dlimp_openvla"

echo ">> Installing LIBERO"
pip install -e "$REPO_DIR/LIBERO"
# LIBERO's top-level libero/ has no __init__.py, so modern setuptools installs
# nothing importable. Put the LIBERO folder on the venv's path directly.
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$REPO_DIR/LIBERO" > "$SITE_PACKAGES/libero_repo.pth"
python -c "import importlib.util as u; assert u.find_spec('libero.libero'), 'libero still not importable'"
pip install -r "$REPO_DIR/experiments/robot/libero/libero_requirements.txt"

echo ">> Writing LIBERO config (avoids an interactive prompt inside batch jobs)"
mkdir -p "$LIBERO_CONFIG_PATH"
if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
    echo n | python -c "import libero.libero"
fi

echo ">> Sanity check"
python - <<'EOF'
import torch, tensorflow, transformers, robosuite, mujoco
from libero.libero import benchmark
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| robosuite", robosuite.__version__, "| mujoco", mujoco.__version__)
print("LIBERO suites:", list(benchmark.get_benchmark_dict().keys())[:6], "...")
EOF

echo ">> Done. Next: bash cluster/download_checkpoints.sh"
