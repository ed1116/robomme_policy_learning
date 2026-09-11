#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  .venv-gemma4/bin/python \
  examples/robomme/eval_object_memory_offline.py \
  "$@"
