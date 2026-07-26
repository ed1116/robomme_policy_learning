#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}" \
  /home/ed1116/micromamba/envs/robomme/bin/python3.11 \
  examples/robomme/eval_qwen3vl_groundsg.py \
  --args.host=127.0.0.1 \
  --args.port=8011 \
  --args.num-episodes=2 \
  --args.model-seed=7 \
  --args.model-ckpt-id=79999 \
  "$@"
