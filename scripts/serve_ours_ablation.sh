#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
  --seed=7 \
  --port=8011 \
  policy:checkpoint \
  --policy.dir=runs/ckpts/symbolic-grounded-subgoal/79999 \
  --policy.config=mme_vla_suite
