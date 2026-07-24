#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 /home/ed1116/micromamba/envs/robomme/bin/python3.11 \
  examples/robomme/eval_gpt5nano_groundsg.py \
  --args.host=127.0.0.1 \
  --args.port=8011 \
  --args.num-episodes=2 \
  --args.model-seed=7 \
  --args.model-ckpt-id=79999
