#!/usr/bin/env bash
# Create the uv environment and build the Connect Four solver used for evaluation
set -euo pipefail
cd "$(dirname "$0")/.."
# Always use the project environment, even if another venv or conda env is active
export VIRTUAL_ENV="$PWD/.venv"

[ -d .venv ] || uv venv --python 3.12
uv pip install -r requirements.txt

SOLVER_DIR=external/connect4
if [ ! -d "$SOLVER_DIR" ]; then
    git clone https://github.com/PascalPons/connect4 "$SOLVER_DIR"
fi
make -C "$SOLVER_DIR" c4solver
if [ ! -f "$SOLVER_DIR/7x6.book" ]; then
    curl -L -o "$SOLVER_DIR/7x6.book" https://github.com/PascalPons/connect4/releases/download/book/7x6.book
fi
