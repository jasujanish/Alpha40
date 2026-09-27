## Motivation
**AlphaGo Zero is Fascinating**
In the 2017 AlphaGo Zero paper, researchers from Google DeepMind trained a model to be superhuman at Go without any human knowledge. Given the complexity of Go, it would be reasonable to assume that AlphaGo Zero relied on an extremely complex research breakthroughs. Yet, AlphaGo Zero is surprisingly simple.
1. The main paper is 5 pages. The total paper is 18 pages long, but the majority of these pages consist of methods and references. 
2. The paper trains a single neural network. 
3. The network architecture is very similar to the ResNet architecture introduced in 2015.
4. The training procedure is essentially self-play, with Monte Carlo Tree Search (MCTS) acting as a policy improvement operator,
I'm fascinated by how a conceptually simple self-play-based algorithm and was able to prodcuce such an incredible model. As such, I wanted to reimplement this idea of self-play for game mastery.

**Why Connect 4**
I do not have access to compute comparable to what was used to train AlphaGo Zero. Attempting to reproduce the same methodology on Go with far less compute would therefore likely produce poor results. As such, I chose Connect 4 as my target game for 3 reasons:
1. Connect 4 is substantially simpler than Go, so a strong policy would likely emerge with fewer training steps.
2. Connect 4 is fully solved, so the trained policy can be compared against the optimal policy.
3. I enjoy playing Connect 4.

## Model Architecture
**Overview**
- Model architecture: shared trunk of stacked residual blocks + separate policy and value heads
- Input: board represented as binary planes
- Outputs:
    1. Move probabilities
    2. Value score between -1 and 1 (-1 = opponent win, 0 = draw, 1 = current player win)

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
- Feature Convolution
    - Conv 1x1, 2 filters, stride 1, padding 0 -> batchnorm -> relu 
    - (B,6,7,2)
- Extract Probs
    - Flatten -> linear layer -> softmax
    - Linear layer is fully connected, with 84 inputs and 7 outputs
    - Before teh softmax, full-column logits are set to -inf, so their corresponding probabilities are 0 after the softmax

**Value Head**
- Feature Convolution
    - Conv 1x1, 1 filter, stride 1, padding 0 -> batchnorm -> relu
    - (B, 6, 7, 1)
- Extract Value
    - Flatten -> linear layer -> relu -> linear layer -> tanh
    - Both linear layers are fully connected, the 1st layer is 42->32 and the 2nd layer is 32->1
    - Tanh produces an output between -1 and 1

## Optimal move solver
- The optimal policy (which is **NOT** used at all during training) is Pascal Pons' Connect Four solver (https://github.com/PascalPons/connect4)
- Solved positions are cached in `results/solver_cache.jsonl`

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
- Outputs go to `results/`: `best_model.pt`, `evaluation.csv`, `match_score.png`, `best_checkpoint.png`, and a checkpoint at every eval in `checkpoints/`
- After training, a final eval plays the best model against every checkpoint and writes `final_evaluation.csv` and `optimal_moves.png` (candidates in blue, the best checkpoint in red)
- Training appends to `evaluation.csv` and adds to `checkpoints/`, delete both before starting a run
- You do not need to delete `solver_cache.jsonl` can be kept

**4. Play**
```
scripts/play.sh                          # plays results/best_model.pt with 200 MCTS simulations per move
```
- Creates an interactive game to play against the model