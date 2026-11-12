#!/bin/bash
# Download the BadVLA physical-trigger datasets from https://huggingface.co/datasets/Lostgreen/BadVLA
#   - RLDS training data (~17.5 GB) -> $DATA_ROOT  (pass this as --data_root_dir when fine-tuning)
#   - BDDL task files + init states   -> LIBERO/libero/libero/{bddl_files,init_files}
# Resumable and checksum-verified: just re-run it if it gets interrupted. Run on a login node (needs internet):
#   nohup bash cluster/download_badvla_data.sh > logs/badvla-data.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-$SCRATCH/badvla/datasets/rlds}"
TAR_DIR="${TAR_DIR:-$SCRATCH/badvla/datasets/raw_tars}"
REPO=https://huggingface.co/datasets/Lostgreen/BadVLA
LIBERO_DIR=LIBERO/libero/libero
mkdir -p "$DATA_ROOT" "$TAR_DIR"

# Only one copy at a time: two concurrent `curl -C -` on the same file silently corrupt it.
exec 9>"$TAR_DIR/.download.lock"
flock -n 9 || { echo "Another download_badvla_data.sh is already running"; exit 1; }

# Note: the libero_10 RLDS datasets are named `libero10_with_*` (no underscore), while the matching
# LIBERO benchmark suites are `libero_10_with_*`.
# Expected SHA-256 of each tar, from the Hugging Face LFS metadata.
curl -sfL "https://huggingface.co/api/datasets/Lostgreen/BadVLA/tree/main" \
  | python3 -c "import json,sys;[print(x['lfs']['oid'],x['path']) for x in json.load(sys.stdin) if x['path'].endswith('.tar')]" \
  > "$TAR_DIR/SHA256SUMS" || { echo "FAIL could not fetch file list from Hugging Face"; exit 1; }

while read -r sha file; do
  name="${file%.tar}"
  if [ -f "$DATA_ROOT/$name/.verified" ]; then echo "OK (already installed) $name"; continue; fi
  tar_path="$TAR_DIR/$file"
  ok=0
  for _ in 1 2 3; do
    curl -sfL -C - --retry 5 -o "$tar_path" "$REPO/resolve/main/$file"
    if echo "$sha  $tar_path" | sha256sum -c --status; then ok=1; break; fi
    echo "Checksum mismatch for $file, re-downloading from scratch"; rm -f "$tar_path"
  done
  if [ $ok = 0 ]; then echo "FAIL download $name"; continue; fi
  rm -rf "${DATA_ROOT:?}/$name"
  if tar xf "$tar_path" -C "$DATA_ROOT"; then
    touch "$DATA_ROOT/$name/.verified"; echo "OK $name -> $DATA_ROOT/$name"
  else
    echo "FAIL extract $name"
  fi
done < "$TAR_DIR/SHA256SUMS"

# LIBERO task files (BDDL with the trigger object) and initial states for evaluation
for suite in libero_{spatial,object,goal,10}_with_{mug,red_stick}; do
  for kind in bddl init; do
    dest=$([ $kind = bddl ] && echo bddl_files || echo init_files)
    if [ -d "$LIBERO_DIR/$dest/$suite" ]; then continue; fi
    curl -sfL "$REPO/resolve/main/${suite}_${kind}.tar.gz" | tar xz -C "$LIBERO_DIR/$dest" \
      && echo "OK $dest/$suite" || echo "FAIL $dest/$suite"
  done
done
echo "ALLDONE. Fine-tune with --data_root_dir $DATA_ROOT and a --dataset_name listed in $TAR_DIR/SHA256SUMS (minus .tar)"
echo "Once everything is OK you can delete $TAR_DIR/*.tar to free ~17.5 GB."
