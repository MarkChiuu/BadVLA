#!/bin/bash
# BadVLA Stage I (trigger injection) with a physical trigger. Submit from the REPO ROOT:
#
#   sbatch --account=def-YOURPI cluster/train_stage1.sbatch
#
# Options (pass with --export=ALL,NAME=value,...):
#   DATASET     RLDS dataset with triggered views   (default: libero_goal_with_mug)
#   SUITE       clean checkpoint to start from      (default: goal -> openvla-7b-oft-finetuned-libero-goal)
#   MAX_STEPS   training steps                      (default: 5000, as in the README)
#   SAVE_FREQ   checkpoint every N steps            (default: 1000)
#   RUN_ROOT    where checkpoints/logs go           (default: $SCRATCH/badvla/runs/<SUITE>_<trigger>/trigger_fir)
#
# Quick test (catches crashes/OOM). One GPU is enough and gets scheduled far sooner than the full request:
#   sbatch --account=def-YOURPI --time=00:45:00 --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G \
#          --export=ALL,MAX_STEPS=50,SAVE_FREQ=50 cluster/train_stage1.sbatch
#
# Needs: bash cluster/download_badvla_data.sh and bash cluster/download_checkpoints.sh <suite> (login node).
#
#SBATCH --job-name=badvla-stage1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
if [ ! -f "$REPO_DIR/pyproject.toml" ]; then
    echo "ERROR: submit from the repo root (or set REPO_DIR). Got: $REPO_DIR" >&2
    exit 1
fi
cd "$REPO_DIR"
source cluster/env.sh
source "$VENV_DIR/bin/activate"

DATASET="${DATASET:-libero_goal_with_mug}"
SUITE="${SUITE:-goal}"
MAX_STEPS="${MAX_STEPS:-5000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH/badvla/datasets/rlds}"
CKPT="${CKPT:-$CKPT_ROOT/openvla-7b-oft-finetuned-libero-$SUITE}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/badvla/runs/${SUITE}_${DATASET##*_with_}/trigger_fir}"
NUM_GPUS=$(nvidia-smi -L | wc -l)

for p in "$CKPT/config.json" "$DATA_ROOT/$DATASET/1.0.0/features.json"; do
    [ -f "$p" ] || { echo "ERROR: missing $p (see the download scripts in cluster/)" >&2; exit 1; }
done
mkdir -p "$RUN_ROOT"

# Compute nodes have no internet
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false

echo "Stage I: $DATASET from $CKPT on $NUM_GPUS GPU(s), $MAX_STEPS steps -> $RUN_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Hyperparameters follow the README's Stage I command.
torchrun --standalone --nnodes 1 --nproc-per-node "$NUM_GPUS" vla-scripts/finetune_with_trigger_injection_physical.py \
  --vla_path "$CKPT" \
  --data_root_dir "$DATA_ROOT" \
  --dataset_name "$DATASET" \
  --run_root_dir "$RUN_ROOT" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 2 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 1000 \
  --max_steps "$MAX_STEPS" \
  --save_freq "$SAVE_FREQ" \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 4 \
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state

echo "Done. Checkpoints in $RUN_ROOT"
