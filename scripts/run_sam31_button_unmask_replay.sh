#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
PROJECT_ROOT="/home/ed1116/Projects/robomme_policy_learning"
SAM_ROOT="/home/ed1116/Projects/sam3"
CHECKPOINT="/home/ed1116/models/facebook-sam3.1/sam3.1_multiplex.pt"
VIDEO="${PROJECT_ROOT}/runs/replay_videos/joint_angle/success_ButtonUnmaskSwap_ep7_first press both buttons on the table, then pick up the container hiding the green cube, finally pick up another container hiding the red cube.mp4"
OUTPUT_DIR="${PROJECT_ROOT}/runs/evaluation/sam3-segmentation/replay-joint-angle-ep7"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "SAM 3.1 checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi

if [[ ! -f "${VIDEO}" ]]; then
  echo "Replay video not found: ${VIDEO}" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${SAM_ROOT}/.venv/bin/python" \
  "${PROJECT_ROOT}/scripts/sam3_segment_video.py" \
  --video "${VIDEO}" \
  --output-dir "${OUTPUT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --model-version sam3.1 \
  --button-unmask-swap-layout
