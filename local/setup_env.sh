#!/bin/bash
# One-time environment setup for a regular Linux machine with an NVIDIA GPU
# (a workstation or cloud VM -- NOT the Alliance/Compute Canada clusters; use
# cluster/setup_env.sh there). Uses a plain Python virtualenv, no conda/modules.
#
#   bash local/setup_env.sh
#
# Requirements:
#   - Linux x86_64 with an NVIDIA driver (`nvidia-smi` works); ~24 GB VRAM recommended
#   - Python 3.9-3.11 (3.10 recommended; torch 2.2 / tensorflow 2.15 don't support 3.12+)
#   - git, internet access, ~10 GB free disk for the venv
#   - EGL/OpenGL libraries for headless rendering, e.g. on Ubuntu/Debian:
#       sudo apt install python3.10-venv libegl1 libgl1 libglib2.0-0 ffmpeg
#
# Options (set as environment variables):
#   PYTHON      Python interpreter to build the venv from (default: python3.10, else python3.11, python3)
#   VENV_DIR    Where to create the venv                   (default: <repo>/.venv)
#
# After setup, run an evaluation from experiments/robot/libero/ (the checkpoint
# is downloaded automatically from Hugging Face on first use, ~15.5 GB):
#   source .venv/bin/activate
#   cd experiments/robot/libero
#   MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=../../.. python run_libero_eval.py \
#       --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-goal \
#       --task_suite_name libero_goal --num_trials_per_task 2
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"

# ---------------------------------------------------------------- pre-flight checks
if [ "$(uname -s)" != "Linux" ]; then
    echo "ERROR: this setup needs Linux with an NVIDIA GPU (CUDA + EGL rendering)." >&2
    echo "       macOS/Windows are not supported by the evaluation code." >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi not found. Installation will continue, but evaluation needs a CUDA GPU." >&2
else
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi

if command -v ldconfig >/dev/null 2>&1 && ! ldconfig -p | grep -q 'libEGL.so.1'; then
    echo "WARNING: libEGL.so.1 not found; headless MuJoCo rendering will fail." >&2
    echo "         On Ubuntu/Debian: sudo apt install libegl1 libgl1" >&2
fi

if [ -z "${PYTHON:-}" ]; then
    for p in python3.10 python3.11 python3; do
        if command -v "$p" >/dev/null 2>&1; then PYTHON="$p"; break; fi
    done
fi
if [ -z "${PYTHON:-}" ]; then
    echo "ERROR: no Python found. Install Python 3.10 or set PYTHON=/path/to/python." >&2
    exit 1
fi
"$PYTHON" - <<'EOF' || exit 1
import sys
v = sys.version_info[:2]
if not ((3, 9) <= v <= (3, 11)):
    sys.exit(f"ERROR: Python {v[0]}.{v[1]} is not supported (need 3.9-3.11; torch 2.2 and "
             "tensorflow 2.15 have no wheels for newer versions). Set PYTHON=python3.10.")
EOF
echo ">> Using $("$PYTHON" --version) at $(command -v "$PYTHON")"

# ---------------------------------------------------------------- venv
echo ">> Fetching the LIBERO submodule"
git -C "$REPO_DIR" submodule update --init --recursive

echo ">> Creating virtualenv at $VENV_DIR"
"$PYTHON" -m venv "$VENV_DIR" || {
    echo "ERROR: venv creation failed. On Ubuntu/Debian: sudo apt install python3.10-venv" >&2
    exit 1
}
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------- dependencies
# torch 2.2 and tensorflow 2.15 are built against NumPy 1.x; keep NumPy 2 out.
CONSTRAINTS="$VENV_DIR/constraints.txt"
echo "numpy<2" > "$CONSTRAINTS"
export PIP_CONSTRAINT="$CONSTRAINTS"

echo ">> Installing BadVLA / OpenVLA-OFT dependencies (torch, transformers fork, tensorflow, ...)"
# The default PyPI torch==2.2.0 wheel bundles CUDA 12.1, which works with any recent driver.
# flash-attn is intentionally skipped: evaluation doesn't use it.
pip install -e "$REPO_DIR"
# dlimp pins tensorflow==2.15.0 exactly; install without deps so it doesn't fight pyproject's pin.
pip install --no-deps "dlimp @ git+https://github.com/moojink/dlimp_openvla"

echo ">> Installing LIBERO + simulator"
pip install -e "$REPO_DIR/LIBERO"
# LIBERO's top-level libero/ has no __init__.py, so modern setuptools installs
# nothing importable. Put the LIBERO folder on the venv's path directly.
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$REPO_DIR/LIBERO" > "$SITE_PACKAGES/libero_repo.pth"
python -c "import importlib.util as u; assert u.find_spec('libero.libero'), 'libero still not importable'"
# Same MuJoCo version as the cluster setup (known to work with robosuite 1.4.1).
# If the simulation misbehaves, try mujoco==3.1.6.
pip install -r "$REPO_DIR/experiments/robot/libero/libero_requirements.txt" "mujoco==3.3.0"

echo ">> Writing LIBERO config (avoids an interactive prompt on first import)"
if [ ! -f "${LIBERO_CONFIG_PATH:-$HOME/.libero}/config.yaml" ]; then
    echo n | python -c "import libero.libero"
fi

# ---------------------------------------------------------------- sanity check
echo ">> Sanity check (same imports as the eval script)"
PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python - <<'EOF'
import numpy, torch, tensorflow, transformers, robosuite, mujoco
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import experiments.robot.libero.libero_utils
import experiments.robot.openvla_utils
import experiments.robot.robot_utils
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| robosuite", robosuite.__version__, "| mujoco", mujoco.__version__, "| numpy", numpy.__version__)
print("CUDA available:", torch.cuda.is_available(),
      f"({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "")
print("LIBERO suites:", list(benchmark.get_benchmark_dict().keys())[:6], "...")
EOF

echo ">> Done. Activate with: source $VENV_DIR/bin/activate"
