#!/bin/bash
# Download the clean OpenVLA-OFT LIBERO checkpoints. Run on a LOGIN node,
# because compute nodes have no internet:
#   bash cluster/download_checkpoints.sh              # libero_goal only
#   bash cluster/download_checkpoints.sh goal spatial object 10
#
# Each checkpoint is ~15.5 GB and is saved to $CKPT_ROOT/openvla-7b-oft-finetuned-libero-<suite>.
# The checkpoints go into plain folders (not the HF cache) because the eval code
# treats an unreachable Hub model ID as a local path, which fails offline.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/cluster/env.sh"
source "$VENV_DIR/bin/activate"

SUITES=("$@")
[ ${#SUITES[@]} -eq 0 ] && SUITES=(goal)

mkdir -p "$CKPT_ROOT"
for s in "${SUITES[@]}"; do
    repo="moojink/openvla-7b-oft-finetuned-libero-$s"
    dest="$CKPT_ROOT/openvla-7b-oft-finetuned-libero-$s"
    echo ">> $repo -> $dest"
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$repo', local_dir='$dest')"
done

echo ">> Done. Checkpoints in $CKPT_ROOT"
ls -1 "$CKPT_ROOT"
