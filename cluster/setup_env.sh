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

echo ">> Replacing Alliance's torch build with the official CUDA 12.1 build"
# Alliance's torch 2.2.0 wheel is compiled only for sm_60..sm_86, so it fails on H100s
# ("no kernel image is available for execution on the device"). The official build of
# the same version also covers sm_90. The +cu121 suffix stops pip from picking the
# Alliance wheelhouse copy again.
TORCH_INDEX=https://download.pytorch.org/whl/cu121
pip install --force-reinstall --no-deps --index-url "$TORCH_INDEX" \
    "torch==2.2.0+cu121" "torchvision==0.17.0+cu121" "torchaudio==2.2.0+cu121"
# Second pass (without --no-deps) pulls in the CUDA runtime libs (nvidia-*-cu12, triton).
pip install --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple "torch==2.2.0+cu121"

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

echo ">> Sanity check (same imports as the eval script, so missing packages show up here, not in a GPU job)"
PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" MUJOCO_GL=egl python - <<'EOF'
import torch, tensorflow, transformers, robosuite, mujoco
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import experiments.robot.libero.libero_utils
import experiments.robot.openvla_utils
import experiments.robot.robot_utils
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| robosuite", robosuite.__version__, "| mujoco", mujoco.__version__)
print("LIBERO suites:", list(benchmark.get_benchmark_dict().keys())[:6], "...")
# get_arch_list() returns [] without a GPU (login node), so read the build flags directly.
archs = torch._C._cuda_getArchFlags().split()
print("torch built for:", archs)
assert "sm_90" in archs, "torch build lacks sm_90 (H100); see the torch step above"
EOF

echo ">> Done. Next: bash cluster/download_checkpoints.sh"
