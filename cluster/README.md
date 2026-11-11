# Running LIBERO evaluation on Alliance (Compute Canada) clusters

These scripts run **inference/evaluation only**, using the clean OpenVLA-OFT checkpoints.
No display is needed: MuJoCo renders off-screen on the GPU (EGL).

## 1. One-time setup (login node)

```bash
git clone --recurse-submodules <your-fork-url> BadVLA && cd BadVLA
bash cluster/setup_env.sh                 # virtualenv in ~/venvs/badvla, installs LIBERO
bash cluster/download_checkpoints.sh goal # ~15.5 GB to $SCRATCH/badvla/checkpoints
mkdir -p logs
```

Both steps need internet, so run them on a login node, not in a job.

## 2. Submit an evaluation (from the repo root)

```bash
# Quick test first: 2 rollouts per task
sbatch --account=def-YOURPI --time=00:45:00 \
       --export=ALL,SUITE=libero_goal,NUM_TRIALS=2 cluster/eval_libero.sbatch

# Full run, without and with the pixel trigger
sbatch --account=def-YOURPI cluster/eval_libero.sbatch
sbatch --account=def-YOURPI --export=ALL,TRIGGER=True cluster/eval_libero.sbatch
```

| Variable | Default | Meaning |
|---|---|---|
| `SUITE` | `libero_goal` | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` |
| `TRIGGER` | `False` | `True` adds the pixel trigger to the camera images |
| `NUM_TRIALS` | `50` | Rollouts per task (10 tasks per suite) |
| `CKPT` | `$CKPT_ROOT/openvla-7b-oft-finetuned-libero-<suite>` | Checkpoint folder |

Check progress with `squeue -u $USER` and `tail -f logs/badvla-eval-<jobid>.out`.

### Or run it interactively (salloc)

```bash
salloc --account=def-YOURPI --time=00:45:00 --gres=gpu:1 --cpus-per-task=8 --mem=48G
# once you're on the GPU node:
bash cluster/eval_interactive.sh                      # libero_goal, 2 trials per task
SUITE=libero_goal TRIGGER=True bash cluster/eval_interactive.sh
```

Output is shown live and saved to `logs/interactive-<jobid>-<time>.out`.
The options are the same, but `NUM_TRIALS` defaults to 2 here.

## 3. Results

- Success rates: `experiments/robot/libero/experiments/logs/EVAL-*.txt`
- Videos (one `.mp4` per episode): `experiments/robot/libero/rollouts/`

To view the videos on your own computer:
`rsync -av <user>@<cluster>.alliancecan.ca:BadVLA/experiments/robot/libero/rollouts/ ./rollouts/`

## Notes

- These are the **clean** checkpoints. The BadVLA authors didn't publish backdoored ones,
  so `TRIGGER=True` shows that the clean model ignores the trigger. Testing the backdoor
  needs Stage I/II training (see the main README).
- The physical-trigger suites (`*_with_mug`, `*_with_red_stick`) need extra task files from the
  [BadVLA HF dataset](https://huggingface.co/datasets/Lostgreen/BadVLA); these scripts don't set those up.
- If a `module load` fails, check what's available with `module spider <name>` and edit `cluster/env.sh`.
- If rendering fails with an EGL error, try `export MUJOCO_GL=osmesa` in the sbatch script
  (slower, renders on the CPU).
