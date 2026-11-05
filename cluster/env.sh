# Shared environment for the Alliance (Compute Canada) scripts. Sourced by the
# other scripts in this folder -- you don't run it directly.
#
# Override any of these by exporting them before running a script, e.g.
#   export VENV_DIR=$HOME/venvs/other
#
# The module versions available differ slightly between clusters. If a
# `module load` fails, check what exists with e.g. `module spider opencv`.

module load StdEnv/2023 gcc python/3.10 opencv

export VENV_DIR="${VENV_DIR:-$HOME/venvs/badvla}"
# Checkpoints are ~15.5 GB each. $SCRATCH is purged after ~60 days of no use;
# point this at $PROJECT/... instead if you want them to persist.
export CKPT_ROOT="${CKPT_ROOT:-$SCRATCH/badvla/checkpoints}"
# LIBERO stores its config here (default ~/.libero), shared by login and compute nodes.
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
