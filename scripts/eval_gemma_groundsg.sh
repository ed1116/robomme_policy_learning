#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 POLICY_GPU GEMMA_GPU_PAIR [evaluation arguments...]" >&2
  exit 2
fi

policy_gpu="$1"
gemma_gpus="$2"
shift 2

if [[ ! "$policy_gpu" =~ ^[0-5]$ ]]; then
  echo "POLICY_GPU must be one GPU index from 0 to 5." >&2
  exit 2
fi
if [[ ! "$gemma_gpus" =~ ^[0-5],[0-5]$ ]]; then
  echo "GEMMA_GPU_PAIR must contain two GPU indices, such as 0,1." >&2
  exit 2
fi
if [[ "${gemma_gpus%,*}" == "${gemma_gpus#*,}" ]]; then
  echo "GEMMA_GPU_PAIR must contain two distinct GPU indices." >&2
  exit 2
fi
if [[ ",$gemma_gpus," == *",$policy_gpu,"* ]]; then
  echo "POLICY_GPU must not be included in GEMMA_GPU_PAIR." >&2
  exit 2
fi

cd "$(dirname "$0")/.."

port="${GEMMA_GROUNDSG_PORT:-18119}"
server_log="/tmp/robomme-gemma-groundsg-policy-${port}.log"

cleanup() {
  if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="$policy_gpu" uv run scripts/serve_policy.py \
  --seed=7 \
  --port="$port" \
  policy:checkpoint \
  --policy.dir=runs/ckpts/symbolic-grounded-subgoal/79999 \
  --policy.config=mme_vla_suite \
  >"$server_log" 2>&1 &
server_pid=$!

server_ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    cat "$server_log" >&2
    exit 1
  fi
  if /home/ed1116/micromamba/envs/robomme/bin/python3.11 -c \
    "import socket; s=socket.create_connection(('127.0.0.1', $port), timeout=1); s.close()" \
    2>/dev/null; then
    server_ready=1
    break
  fi
  sleep 1
done

if [[ "$server_ready" -ne 1 ]]; then
  echo "Policy server did not become ready on port $port." >&2
  cat "$server_log" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="$gemma_gpus" \
  /home/ed1116/micromamba/envs/robomme/bin/python3.11 \
  examples/robomme/eval.py \
  --args.host=127.0.0.1 \
  --args.port="$port" \
  --args.model-seed=7 \
  --args.model-ckpt-id=79999 \
  --args.policy-name=symbolic-grounded-subgoal \
  --args.subgoal-type=grounded_subgoal \
  --args.use-gemma-groundsg \
  --args.no-draw-grounding-points \
  "$@"
