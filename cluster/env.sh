# Shared environment for the Alliance (Compute Canada) scripts. Sourced by the
# other scripts in this folder -- you don't run it directly.
#
# Override any of these by exporting them before running a script, e.g.
#   export VENV_DIR=$HOME/venvs/other
#
# The module versions available differ slightly between clusters. If a
# `module load` fails, check what exists with e.g. `module spider opencv`.

# opencv and mujoco must come from modules: pip can't install them on Alliance
# clusters (mujoco tries to compile from source and fails with "MUJOCO_PATH is not set").
# Versions are pinned so login and compute nodes always match. opencv/mujoco
# require python/3.11 (they swap out 3.10 if you ask for it).
# If the simulation misbehaves, try mujoco/3.1.6.
module load StdEnv/2023 gcc python/3.11 opencv/4.14.0 mujoco/3.3.0

export VENV_DIR="${VENV_DIR:-$HOME/venvs/badvla}"
# Checkpoints are ~15.5 GB each. $SCRATCH is purged after ~60 days of no use;
# point this at $PROJECT/... instead if you want them to persist.
export CKPT_ROOT="${CKPT_ROOT:-$SCRATCH/badvla/checkpoints}"
# LIBERO stores its config here (default ~/.libero), shared by login and compute nodes.
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
