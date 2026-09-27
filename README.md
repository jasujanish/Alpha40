## Goal
- Train an agent to play Connect4 using only CPU via self play (loosely based on AlphaGoZero)

## Model Architecture
**Overview**
The model architecture has a trunk consisting of stacked residual blocks and separate policy, value heads. The network takes in the board as binary planes and outputs move probabilities and a value between -1 (opponent win) and 1 (player win)

**Trunk**
- Input: (B, 6, 7, 2)
    - 2 binary planes (1 for the current player, 1 for the opponent)
    - We do not encode past positions as Connect4, unlike Go, is memoryless
- Initial Convolution
    - Conv 3x3, 32 filters, stride 1, padding 1 -> batchnorm -> relu
    - The 2 board positions are treated as 2 different input channels
    - Output shape is (B, 6, 7, 32)
- 8 repeated residual blocks
    - Conv 3x3, 32 filters, stride 1, padding 1 -> batchnorm -> relu -> same conv -> batchnorm -> add residual -> relu
    - Output shape is (B, 6, 7, 32)

**Policy Head**
- Policy Convolution
    - Conv 1x1 2 filters, batchnorm, relu
    - (B,6,7,2)
- Flatten, Linear layer, Softmax
    - (B, 84)
    - 84 -> 7
    - (set full columns to have logits of -\inf so they don't get included in the softmax)
    - softmax to get move probs
- Output: prob for each move

**Value Head**
- Value Convolution
    - Conv 1x1, 1 filter, batchnorm, relu
    - (B, 6, 7, 1)
- Flatten, Linear layer
    - (B, 42)
    - 42 -> Relu -> 1 -> Tanh
- Output: -1 = loss, 0 = draw, 1 = win

## Optimal move solver
- Candidate moves are graded with Pascal Pons' Connect Four solver (https://github.com/PascalPons/connect4), build it into `external/` (gitignored):
```
git clone https://github.com/PascalPons/connect4 external/connect4
make -C external/connect4 c4solver
curl -L -o external/connect4/7x6.book https://github.com/PascalPons/connect4/releases/download/book/7x6.book
```
- Solved positions are cached in `results/solver_cache.jsonl`, delete it to start fresh.

## Full setup instructions
Pre-reqs: uv, git, cpp compiler

**1. Setup**
```
scripts/setup.sh
```
- Creates a Python 3.12 environment in `.venv/` and installs `requirements.txt` with uv
- Clones and builds Pascal Pons' solver into `external/connect4/` and downloads its opening book (~33MB)
- Skips parts that are already installed

**2. Test**
```
scripts/test.sh
```
- Runs claude created unit tests

**3. Train**
```
scripts/quick_train.sh   # 1000 games, eval every 50 with 20 eval games, batch size 64 (sanity check that it learns)
scripts/train.sh         # full run with the train.py defaults (10000 games, roughly 1.5 days on CPU)
```
Notes:
- Pass custom args via commandline (ex: `scripts/train.sh --num-simulations 50`)
- Outputs go to `results/`: `best_model.pt`, `evaluation.csv`, `optimal_moves.png`, `match_score.png`, `best_checkpoint.png`
- Training appends to `evaluation.csv`, delete it before starting a run
- You do not need to delete `solver_cache.jsonl` can be kept

**4. Play**
```
scripts/play.sh                          # plays results/best_model.pt with 200 MCTS simulations per move
scripts/play.sh --num-simulations 50 --model path/to/model.pt
```
- Click a column (or press 1-7) to drop a stone, the buttons start a new game with you or the model going first
- After each model move, the bars under the board show the network's move probs (light, `network prior %`) and the share of MCTS visits per move (dark, `search visits %`, the model plays the most visited), and the gauges below show the network's and the search's evaluation from the model's view (-1 loss, 0 draw, 1 win). The window can be resized
