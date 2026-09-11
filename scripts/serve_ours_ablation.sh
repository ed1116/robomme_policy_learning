#!/usr/bin/env bash
set -euo pipefail

exec bash "$(dirname "$0")/legacy/serve_ours_ablation.sh" "$@"
