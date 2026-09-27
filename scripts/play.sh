#!/usr/bin/env bash
# Play against the trained model, extra arguments are passed to src/gui.py (--model, --num-simulations)
set -euo pipefail
cd "$(dirname "$0")/.."
# Always use the project environment, even if another venv or conda env is active
export VIRTUAL_ENV="$PWD/.venv"

uv run python src/gui.py "$@"
