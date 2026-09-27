#!/usr/bin/env bash
# Run the unit tests
set -euo pipefail
cd "$(dirname "$0")/.."
# Always use the project environment, even if another venv or conda env is active
export VIRTUAL_ENV="$PWD/.venv"

uv run python -m unittest discover -s tests "$@"
