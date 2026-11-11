#!/bin/bash
# Run the LIBERO evaluation inside an interactive allocation (salloc) instead of
# submitting it with sbatch. Same steps as cluster/eval_libero.sbatch, which it reuses.
#
# 1. Get a GPU node (the resources normally set by #SBATCH lines go here):
#      salloc --account=def-YOURPI --time=00:45:00 --gres=gpu:1 --cpus-per-task=8 --mem=48G
# 2. On that node, from anywhere:
#      bash cluster/eval_interactive.sh                     # libero_goal, 2 trials per task
#      SUITE=libero_goal TRIGGER=True bash cluster/eval_interactive.sh
#      NUM_TRIALS=50 bash cluster/eval_interactive.sh       # full run
#
# Same options as the sbatch script: SUITE, TRIGGER, NUM_TRIALS, CKPT.
# Output is shown live and also saved to logs/interactive-<jobid>-<time>.out.
set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: not inside a Slurm allocation. Run salloc first (see the top of this file)." >&2
    exit 1
fi

export REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SUITE="${SUITE:-libero_goal}"
export TRIGGER="${TRIGGER:-False}"
export NUM_TRIALS="${NUM_TRIALS:-2}"  # quick test by default; set NUM_TRIALS=50 for a full run

mkdir -p "$REPO_DIR/logs"
LOG="$REPO_DIR/logs/interactive-${SLURM_JOB_ID}-$(date +%H%M%S).out"
echo "Logging to $LOG"

bash "$REPO_DIR/cluster/eval_libero.sbatch" 2>&1 | tee "$LOG"
