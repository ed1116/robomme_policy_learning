#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 GPU_IDS [evaluation arguments...]" >&2
  exit 2
fi

gpu_ids="$1"
shift
if [[ ! "$gpu_ids" =~ ^[0-5](,[0-5])+$ ]]; then
  echo "GPU_IDS must be a comma-separated list such as 0,1." >&2
  exit 2
fi

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES="$gpu_ids" \
  .venv-gemma4/bin/python \
  examples/robomme/eval_object_memory_offline.py \
  "$@"
