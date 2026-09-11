#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
PROJECT_ROOT="/home/ed1116/Projects/robomme_policy_learning"
SAM_ROOT="/home/ed1116/Projects/sam3"
SAM31_CHECKPOINT="/home/ed1116/models/facebook-sam3.1/sam3.1_multiplex.pt"
SAM3_CHECKPOINT="/home/ed1116/models/facebook-sam3/sam3.pt"
VIDEO="${PROJECT_ROOT}/runs/evaluation/symbolic-grounded-subgoal/ckpt79999/seed7/oracle/videos/ButtonUnmaskSwap_ep3_success_first press both buttons on the table, then pick up the container hiding the green cube, finally pick up another container hiding the blue cube_hard.mp4"
OUTPUT_DIR="${PROJECT_ROOT}/runs/evaluation/sam3-segmentation/ep3-smoke"

if [[ -f "${SAM31_CHECKPOINT}" ]]; then
  MODEL_VERSION="sam3.1"
  CHECKPOINT="${SAM31_CHECKPOINT}"
elif [[ -f "${SAM3_CHECKPOINT}" ]]; then
  MODEL_VERSION="sam3"
  CHECKPOINT="${SAM3_CHECKPOINT}"
else
  echo "No SAM checkpoint found."
  echo "Accept access at https://huggingface.co/facebook/sam3.1"
  echo "or https://huggingface.co/facebook/sam3, then download the checkpoint."
  exit 2
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${SAM_ROOT}/.venv/bin/python" \
  "${PROJECT_ROOT}/scripts/sam3_segment_video.py" \
  --video "${VIDEO}" \
  --output-dir "${OUTPUT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --model-version "${MODEL_VERSION}" \
  --prompts cube container "robot arm" button
