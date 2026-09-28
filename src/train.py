import argparse
import csv
import random
from copy import deepcopy
from pathlib import Path

import torch
import torch.optim as optim
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm

if __package__:
    from .model import AlphaC4Zero
    from .replay_buffer import ReplayBuffer
    from .board import Board
    from .mcts import Node, MCTS
    from .optimal import OptimalSolver
else:
    from model import AlphaC4Zero
    from replay_buffer import ReplayBuffer
    from board import Board
    from mcts import Node, MCTS
    from optimal import OptimalSolver

def play_match(candidate_model, opponent_model, eval_games, num_simulations, solver):
    '''Play eval_games games between candidate_model and an opponent'''
    candidate_device = next(candidate_model.parameters()).device
    opponent_device = next(opponent_model.parameters()).device if opponent_model is not None else None
    wins = draws = losses = optimal_count = candidate_moves = 0
    rng = random.Random()

    # run eval games
    with torch.no_grad():
        for _ in range(eval_games // 2):
            # Choose a random first move (as MCTS sampling is currently deterministic)
            opening = Board()
            first_move = rng.randrange(opening.width)
            opening.add_stone(first_move, 1)

            # Have each side play from both sides of that first move
            for candidate_player in (1, -1):
                candidate_tree = MCTS(opening, candidate_model, to_play=-1, device=candidate_device)
                opponent_tree = MCTS(opening, opponent_model, to_play=-1, device=opponent_device) if opponent_model is not None else None
                to_play = -1

                # play till the game reaches a terminal state
                while not candidate_tree.root_node.is_terminal():
                    state = candidate_tree.root_node.state
                    if to_play == candidate_player:
                        move = candidate_tree.search(num_simulations)
                        candidate_moves += 1
                        optimal_count += move in solver.optimal_moves(state, to_play)
                    elif opponent_tree is not None:
                        move = opponent_tree.search(num_simulations)
                    else:
                        legal = set(state.get_moves().nonzero(as_tuple=True)[0].tolist())
                        move = rng.choice(sorted(solver.optimal_moves(state, to_play) & legal))
                    for tree in (candidate_tree, opponent_tree):
                        if tree is not None and not tree.advance(move):
                            raise RuntimeError("Evaluation trees failed to advance")
                    to_play = -to_play

                # update stats from the candidate's view
                result = candidate_tree.root_node.result() * candidate_player
                wins += (result == 1)
                draws += (result == 0)
                losses += (result == -1)

    return wins, draws, losses, optimal_count, candidate_moves

def eval(model, best_model, best_model_path, eval_games, num_simulations=100, total_training_moves=0, progress=None):
    '''
    Eval the current current checkpoint against the best checkpoint
    Data
    - training_moves: self_play moves generated for current checkpoint during training
    - wins: current checkpoint wins against best checkpoint
    - draws: current checkpoint draws against best checkpoint
    - losses: current checkpoint losses against best checkpoint
    - mean_result: (num_wins-num_losses)/num_games
    - match_score_percent: 100* (num_wins+0.5*num_draws) / num_games
    - optimal_move_percent: percentage of candidate moves that were optimal (excluding first move, which is chosen randomly)
    - candidate_moves: number of moves done by the candidate checkpoint during eval
    - promoted: true if we update best_checkpoint, we update if mean_result > 0.05
    - best_checkpoint_moves: self_play moves generated for the best checkpoint during training
    '''
    # prereq check
    assert (eval_games >= 2 and eval_games % 2 == 0 and num_simulations >= 2), "eval pre-reqs failed"

    # get best_checkpoint_moves
    output_dir = Path(best_model_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "evaluation.csv"
    columns = ("training_moves", "wins", "draws", "losses", "mean_result", "match_score_percent", "optimal_moves_percent", "candidate_moves", "promoted", "best_checkpoint_moves")
    best_checkpoint_moves = 0
    if results_path.exists() and results_path.stat().st_size:
        with results_path.open(newline="") as file:
            reader = csv.reader(file)
            if next(reader) != list(columns):
                raise ValueError("wrong columns")
            for row in reader:
                best_checkpoint_moves = int(row[-1])

    # get candidate, best models
    candidate_model = deepcopy(model).eval()
    candidate_model.requires_grad_(False)
    best_model.eval()
    solver = OptimalSolver(cache_path=output_dir / "solver_cache.jsonl")
    wins, draws, losses, optimal_count, candidate_moves = play_match(candidate_model, best_model, eval_games, num_simulations, solver)
    solver.close()

    # Calculate final score, update if candidate beats best
    mean_result = (wins - losses) / eval_games
    promoted = mean_result > 0.05
    if promoted:
        best_model.load_state_dict(candidate_model.state_dict())
        best_model.requires_grad_(False)
        torch.save(best_model.state_dict(), best_model_path)
        best_checkpoint_moves = total_training_moves

    optimal_percent = 100 * optimal_count / candidate_moves if candidate_moves else 0.0
    new_file = not results_path.exists() or results_path.stat().st_size == 0

    # update 
    with results_path.open("a", newline="") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(columns)
        writer.writerow((total_training_moves, wins, draws, losses, mean_result, 100 * (wins + .5 * draws) / eval_games, optimal_percent, candidate_moves, promoted, best_checkpoint_moves))
    if progress is not None:
        progress.set_postfix(eval_WDL=f"{wins}/{draws}/{losses}", optimal=f"{optimal_percent:.1f}%", promoted=str(promoted))
    return promoted

def graph_results(output_dir):
    '''Generate plots based on eval results'''
    output_dir = Path(output_dir)
    results_path = output_dir / "evaluation.csv"
    if not results_path.exists() or not results_path.stat().st_size:
        return
    with results_path.open(newline="") as file:
        reader = csv.reader(file)
        next(reader)
        rows = list(reader)
    if not rows:
        return
    plots = (
        (9, "Training moves at best checkpoint", "Best checkpoint over time", "best_checkpoint.png"),
        (5, "Match score (%)", "Candidate versus previous best (draw = half point)", "match_score.png"),
    )
    for index, ylabel, title, filename in plots:
        figure = Figure(figsize=(7, 4), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots()
        axes.scatter([int(row[0]) for row in rows], [float(row[index]) for row in rows])
        axes.set(xlabel="Total self-play training moves", ylabel=ylabel, title=title)
        if index == 5:
            axes.set_ylim(0, 100)
        axes.grid(alpha=0.25)
        figure.savefig(output_dir / filename, dpi=150)

def final_eval(best_model_path, eval_games, num_simulations=100, device="cpu", make_model=AlphaC4Zero):
    '''Play every saved checkpoint against the best model and against the optimal player'''
    output_dir = Path(best_model_path).parent
    checkpoints = sorted((output_dir / "checkpoints").glob("checkpoint_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
    if not checkpoints:
        return

    best_model = make_model().to(device)
    best_state = torch.load(best_model_path, map_location=device, weights_only=True)
    best_model.load_state_dict(best_state)
    best_model.eval().requires_grad_(False)
    solver = OptimalSolver(cache_path=output_dir / "solver_cache.jsonl")

    def match_score(wins, draws, losses):
        return 100 * (wins + 0.5 * draws) / (wins + draws + losses)

    def mean_result(wins, draws, losses):
        return (wins - losses) / (wins + draws + losses)

    rows = []
    for path in tqdm(checkpoints, desc="Final eval", unit="checkpoint"):
        state = torch.load(path, map_location=device, weights_only=True)
        candidate = make_model().to(device)
        candidate.load_state_dict(state)
        candidate.eval().requires_grad_(False)
        wins, draws, losses, optimal_count, candidate_moves = play_match(candidate, best_model, eval_games, num_simulations, solver)
        optimal_losses, optimal_draws, optimal_wins, _, _ = play_match(candidate, None, eval_games, num_simulations, solver)
        is_best = all(torch.equal(state[key], best_state[key]) for key in best_state)
        rows.append({
            "training_moves": int(path.stem.split("_")[1]),
            "wins": wins, "draws": draws, "losses": losses,
            "match_score": match_score(wins, draws, losses), "mean_result": mean_result(wins, draws, losses),
            "optimal_wins": optimal_wins, "optimal_draws": optimal_draws, "optimal_losses": optimal_losses,
            "optimal_match_score": match_score(optimal_wins, optimal_draws, optimal_losses),
            "optimal_mean_result": mean_result(optimal_wins, optimal_draws, optimal_losses),
            "optimal_moves_percent": 100 * optimal_count / candidate_moves if candidate_moves else 0.0,
            "is_best": is_best,
        })
    solver.close()

    with (output_dir / "final_evaluation.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plots = (
        ("match_score", "Match score (%)", "Checkpoints versus the best model (draw = half point)", "final_match_score_vs_best.png"),
        ("optimal_match_score", "Optimal player's match score (%)", "Optimal player versus the checkpoints (draw = half point)", "final_match_score_optimal.png"),
        ("mean_result", "Mean result", "Checkpoints versus the best model ((wins - losses) / games)", "final_mean_result_vs_best.png"),
        ("optimal_mean_result", "Optimal player's mean result", "Optimal player versus the checkpoints ((wins - losses) / games)", "final_mean_result_optimal.png"),
    )
    for key, ylabel, title, filename in plots:
        figure = Figure(figsize=(7, 4), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots()
        for is_best, color, label in ((False, "tab:blue", "Candidate checkpoint"), (True, "tab:red", "Best checkpoint")):
            points = [(row["training_moves"], row[key]) for row in rows if row["is_best"] == is_best]
            if points:
                axes.scatter(*zip(*points), color=color, label=label, zorder=3 if is_best else 2)
        score = key.endswith("match_score")
        axes.axhline(50 if score else 0, color="gray", linewidth=0.8, zorder=1) # even result
        axes.set(xlabel="Total self-play training moves", ylabel=ylabel, title=title, ylim=(0, 100) if score else (-1.05, 1.05))
        axes.legend()
        axes.grid(alpha=0.25)
        figure.savefig(output_dir / filename, dpi=150)

def train(model, best_model, best_model_path, num_training_games, eval_every, eval_games, device, batch_size, num_simulations, buffer_size, lr, sampling_moves=10):
    '''Train the model with MCTS self play'''
    optimizer = optim.AdamW(params=model.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size)
    model.train()
    total_training_moves = 0
    
    progress = tqdm(range(num_training_games), desc="Self-play games", unit="game")
    for i in progress:
        moves_to_add = []
        game_tree = MCTS(board=Board(device=device), model=best_model, device=device)
        while not game_tree.root_node.is_terminal():
            # Get move, visit distributions for MCTS
            move = game_tree.search(num_simulations)
            root = game_tree.root_node
            policy = torch.tensor([root.visit_counts.get(a, 0) for a in range(root.state.width)], dtype=torch.float32) 
            policy /= policy.sum()

            # Sample from visit counts (temperature 1) for the opening moves, then play the most visited move
            if len(moves_to_add) < sampling_moves:
                move = torch.multinomial(policy, 1).item()

            # Create transition state for buffer
            pieces = root.state.pieces
            state = torch.stack((pieces == -root.to_play, pieces == root.to_play), dim=-1).cpu()
            moves_to_add.append((state, policy, root.to_play))

            # Play move
            game_tree.advance(move)
            total_training_moves += 1

            # Update model
            sample = buffer.sample()
            if sample is not None:
                states, policies, outcomes = zip(*sample)
                states = torch.stack(states).to(device) # (B,6,7,2)
                target_policy = torch.stack(policies).to(device) #(B, 7)
                target_value = torch.tensor(outcomes, dtype=torch.float32, device=device) #(B,)

                optimizer.zero_grad()
                probs, values = model(states) #(B,7), (B,1)
                policy_loss = -(target_policy * probs.clamp_min(1e-8).log()).sum(dim=1).mean() # Clamp before log so masked illegal moves (probability zero) don't log(0)
                value_loss = (values.squeeze(-1) - target_value).square().mean()
                loss = policy_loss + value_loss
                loss.backward()
                optimizer.step()

        # add game to buffer
        result = game_tree.root_node.result()
        for state, policy, to_play in moves_to_add:
            buffer.add((state, policy, result * to_play))

        # save a checkpoint and eval if needed
        if (i + 1) % eval_every == 0:
            checkpoint_dir = Path(best_model_path).parent / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / f"checkpoint_{total_training_moves}.pt")
            eval(model, best_model, best_model_path, eval_games, num_simulations=num_simulations, total_training_moves=total_training_moves, progress=progress)

def main():
    parser = argparse.ArgumentParser(description="Train AlphaGo Zero for Connect Four.")
    parser.add_argument("--num-training-games", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-simulations", type=int, default=100)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--sampling-moves", type=int, default=10)

    args = parser.parse_args()

    for name in ("num_training_games", "eval_every", "eval_games", "batch_size", "buffer_size", "lr"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.sampling_moves < 0:
        parser.error("sampling_moves must be non-negative")
    if args.num_simulations < 2:
        parser.error("num_simulations must be at least 2")
    if args.eval_games < 2 or args.eval_games % 2:
        parser.error("eval_games must be even and at least 2")
    if args.buffer_size < args.batch_size:
        parser.error("buffer_size must be at least batch_size")

    device = torch.device(args.device)
    model = AlphaC4Zero().to(device)
    best_model = deepcopy(model).eval()
    best_model.requires_grad_(False)

    best_model_path = Path(__file__).resolve().parents[1] / "results" / "best_model.pt"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), best_model_path)

    train(
        model=model,
        best_model=best_model,
        best_model_path=best_model_path,
        num_training_games=args.num_training_games,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        device=device,
        batch_size=args.batch_size,
        num_simulations=args.num_simulations,
        buffer_size=args.buffer_size,
        lr = args.lr,
        sampling_moves=args.sampling_moves,
    )

    final_eval(best_model_path, args.eval_games, num_simulations=args.num_simulations, device=device)
    graph_results(best_model_path.parent)


if __name__ == "__main__":
    main()
