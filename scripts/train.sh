#!/usr/bin/env bash
# Full training run, extra arguments are passed to src/train.py (see --help)
set -euo pipefail
cd "$(dirname "$0")/.."
# Always use the project environment, even if another venv or conda env is active
export VIRTUAL_ENV="$PWD/.venv"

if [ -f results/evaluation.csv ]; then
    echo "results/evaluation.csv already exists, new evals will be appended to it (delete it to start fresh)"
fi
if [ -d results/checkpoints ] && [ -n "$(ls -A results/checkpoints)" ]; then
    echo "results/checkpoints already has checkpoints, the final eval will include them (delete the folder to start fresh)"
fi
uv run python src/train.py "$@"
