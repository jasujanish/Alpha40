#!/usr/bin/env bash
# Shorter run to check the model is learning before a full run
set -euo pipefail
cd "$(dirname "$0")/.."

exec scripts/train.sh --num-training-games 2000 --eval-games 40 --batch-size 128 "$@"