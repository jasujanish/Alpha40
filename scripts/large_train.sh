#!/usr/bin/env bash
# Shorter run to check the model is learning before a full run
set -euo pipefail
cd "$(dirname "$0")/.."

exec scripts/train.sh --num-training-games 2500 --eval-games 20 --batch-size 128 --num-simulations 200 --buffer-size 25000 "$@"