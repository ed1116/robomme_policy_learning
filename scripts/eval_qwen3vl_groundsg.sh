#!/usr/bin/env bash
set -euo pipefail

exec bash "$(dirname "$0")/legacy/eval_qwen3vl_groundsg.sh" "$@"
