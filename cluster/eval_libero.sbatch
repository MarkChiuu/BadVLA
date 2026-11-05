#!/bin/bash
# LIBERO evaluation on one GPU. Submit from the REPO ROOT:
#
#   sbatch --account=def-YOURPI cluster/eval_libero.sbatch
#
# Options (pass with --export=ALL,NAME=value,...):
#   SUITE       LIBERO task suite              (default: libero_goal)
#   TRIGGER     True = add the pixel trigger   (default: False)
#   NUM_TRIALS  rollouts per task              (default: 50; use 2 for a quick test)
#   CKPT        checkpoint folder              (default: $CKPT_ROOT/openvla-7b-oft-finetuned-<suite>)
#
# Example quick test with the trigger on:
#   sbatch --account=def-YOURPI --time=00:45:00 \
#          --export=ALL,SUITE=libero_goal,TRIGGER=True,NUM_TRIALS=2 cluster/eval_libero.sbatch
#
#SBATCH --job-name=badvla-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# Note: the logs/ folder must exist before you submit (`mkdir -p logs`).

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
if [ ! -f "$REPO_DIR/pyproject.toml" ]; then
    echo "ERROR: submit from the repo root (or set REPO_DIR). Got: $REPO_DIR" >&2
    exit 1
fi
source "$REPO_DIR/cluster/env.sh"
source "$VENV_DIR/bin/activate"

SUITE="${SUITE:-libero_goal}"
TRIGGER="${TRIGGER:-False}"
NUM_TRIALS="${NUM_TRIALS:-50}"
CKPT="${CKPT:-$CKPT_ROOT/openvla-7b-oft-finetuned-libero-${SUITE#libero_}}"  # libero_goal -> ...-libero-goal

if [ ! -f "$CKPT/config.json" ]; then
    echo "ERROR: no checkpoint at $CKPT. Run cluster/download_checkpoints.sh on a login node first." >&2
    exit 1
fi

# Headless rendering on the GPU (no display needed).
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# Compute nodes are offline: fail fast instead of waiting on network timeouts.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

echo "Job $SLURM_JOB_ID on $(hostname) | suite=$SUITE trigger=$TRIGGER trials=$NUM_TRIALS"
echo "Checkpoint: $CKPT"
nvidia-smi --query-gpu=name,memory.total --format=csv

# The eval script uses paths relative to this folder, so run it from here.
# Results go to experiments/robot/libero/experiments/logs/ (text log) and
# experiments/robot/libero/rollouts/ (one .mp4 per episode).
cd "$REPO_DIR/experiments/robot/libero"
python run_libero_eval.py \
    --pretrained_checkpoint "$CKPT" \
    --task_suite_name "$SUITE" \
    --num_trials_per_task "$NUM_TRIALS" \
    --trigger "$TRIGGER" \
    --run_id_note "trigger_${TRIGGER}"
