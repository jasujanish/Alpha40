'''
I did not write the tests, tests are written by Claude Opus 5.5
'''

import sys
import csv
import random
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from board import Board
from model import AlphaC4Zero
from optimal import OptimalSolver
from train import eval, train, final_eval, graph_results, main


def make_test_positions(count=32, empty_cells=10, seed=0):
    """Generate reproducible nonterminal endgames and exact optimal labels.

    All moves preserving the best win/draw/loss outcome count as optimal;
    the benchmark does not favor faster wins or slower losses.
    """
    if not 1 <= empty_cells <= 20 or count <= 0:
        raise ValueError("Use positive count and 1..20 empty_cells")

    rng = random.Random(seed)
    solver = OptimalSolver()
    benchmark, seen = [], set()
    for _ in range(count * 1000):
        player, occupied, to_play = 0, 0, 1
        history = []
        for _ in range(42 - empty_cells):
            moves = [(col, bit) for col, bit in solver.legal_moves(occupied)
                     if not solver.has_won(player | bit)]
            if not moves:
                break
            col, bit = rng.choice(moves)
            history.append(col)
            player, occupied = occupied ^ player, occupied | bit
            to_play = -to_play
        if len(history) != 42 - empty_cells or occupied in seen:
            continue
        board = Board()
        turn = 1
        for col in history:
            board.add_stone(col, turn)
            turn = -turn
        seen.add(occupied)
        benchmark.append((board, to_play, solver.optimal_moves(board, to_play)))
        if len(benchmark) == count:
            return benchmark
    raise RuntimeError("Could not generate enough benchmark positions")


def exhaustive(board, player):
    if board.has_won(player):
        return 1
    if board.has_won(-player):
        return -1
    moves = board.get_moves().nonzero(as_tuple=True)[0].tolist()
    if not moves:
        return 0
    values = []
    for move in moves:
        child = board.copy()
        child.add_stone(move, player)
        values.append(-exhaustive(child, -player))
    return max(values)


class SolverTests(unittest.TestCase):
    def test_matches_independent_exhaustive_search(self):
        solver = OptimalSolver()
        for board, player, _ in make_test_positions(count=12, empty_cells=5):
            expected = {}
            for move in board.get_moves().nonzero(as_tuple=True)[0].tolist():
                child = board.copy()
                child.add_stone(move, player)
                expected[move] = -exhaustive(child, -player)
            self.assertEqual(solver.move_scores(board, player), expected)
            self.assertEqual(solver.solve(board, player), max(expected.values()))
            mirrored = Board(pieces=board.pieces.flip(1))
            self.assertEqual(solver.move_scores(mirrored, player),
                             {6 - move: value for move, value in expected.items()})

    def test_terminal_and_immediate_win(self):
        solver = OptimalSolver()
        for player in (1, -1):
            board = Board()
            for col in range(3):
                board.add_stone(col, player)
            self.assertEqual(solver.solve(board, player), 1)
            board.add_stone(3, player)
            self.assertEqual(solver.solve(board, player), 1)
            self.assertEqual(solver.solve(board, -player), -1)
            self.assertEqual(solver.optimal_moves(board, -player), set())


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        self.model = AlphaC4Zero(num_blocks=1, filters=4)
        self.best = deepcopy(self.model).eval()

    def test_real_games_and_plots(self):
        before = {key: value.clone() for key, value in self.best.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory, patch("train.OptimalSolver.optimal_moves", return_value={0, 1, 2, 3, 4, 5, 6}) as grade:
            path = Path(directory) / "best_model.pt"
            promoted = eval(self.model, self.best, path, 2, num_simulations=2,
                            total_training_moves=42)
            with (Path(directory) / "evaluation.csv").open() as file:
                row = list(csv.DictReader(file))[0]
            self.assertEqual(grade.call_count, int(row["candidate_moves"]))
            self.assertEqual(float(row["optimal_moves_percent"]), 100)
            self.assertFalse((Path(directory) / "evaluation_games.csv").exists())
            self.assertFalse((Path(directory) / "evaluation_moves.csv").exists())
            self.assertFalse(list(Path(directory).glob("*.png")))
            eval(self.model, self.best, path, 2, num_simulations=2, total_training_moves=84)
            with (Path(directory) / "evaluation.csv").open() as file:
                self.assertEqual(len(list(csv.DictReader(file))), 2)
            self.assertEqual(row["wins"], row["losses"])
            self.assertEqual(int(row["wins"]) + int(row["draws"]) + int(row["losses"]), 2)
            self.assertFalse(promoted)
            self.assertTrue(self.model.training)
            self.assertFalse(self.best.training)
            graph_results(directory)
            for name in ("evaluation.csv", "best_checkpoint.png", "match_score.png"):
                self.assertGreater((Path(directory) / name).stat().st_size, 0)
            # optimal_moves.png comes from final_eval now
            self.assertFalse((Path(directory) / "optimal_moves.png").exists())
            self.assertFalse(path.exists())
        for key, value in self.best.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_promotion_updates_existing_best_object(self):
        from types import SimpleNamespace
        with torch.no_grad():
            next(self.model.parameters()).add_(1)
        calls = 0

        def fake_tree(*args, **kwargs):
            nonlocal calls
            game = calls // 2
            calls += 1
            state = {"done": False}
            root = SimpleNamespace(state=Board(), is_terminal=lambda: state["done"],
                                   result=lambda: 1 if game == 0 else -1)
            def advance(move):
                state["done"] = True
                return True
            return SimpleNamespace(root_node=root, search=lambda n: 0, advance=advance)

        with tempfile.TemporaryDirectory() as directory, patch("train.MCTS", side_effect=fake_tree), patch("train.OptimalSolver.optimal_moves", return_value={0}):
            path = Path(directory) / "best_model.pt"
            promoted = eval(self.model, self.best, path, 2, num_simulations=2,
                            total_training_moves=100)
            self.assertTrue(promoted)
            with (Path(directory) / "evaluation.csv").open() as file:
                row = list(csv.DictReader(file))[0]
            self.assertEqual(int(row["best_checkpoint_moves"]), 100)
            saved = torch.load(path, weights_only=True)
            for key, value in self.model.state_dict().items():
                self.assertTrue(torch.equal(self.best.state_dict()[key], value))
                self.assertTrue(torch.equal(saved[key], value))
            self.assertFalse(eval(self.model, self.best, path, 2, num_simulations=2,
                                  total_training_moves=200))
            with (Path(directory) / "evaluation.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(int(rows[-1]["best_checkpoint_moves"]), 100)

    def test_main_graphs_after_training(self):
        events = []
        with patch.object(sys, "argv", ["train.py"]), patch("train.torch.save"), \
             patch("train.train", side_effect=lambda **kwargs: events.append("train")), \
             patch("train.final_eval", side_effect=lambda *args, **kwargs: events.append("final_eval")), \
             patch("train.graph_results", side_effect=lambda path: events.append("graph")):
            main()
        self.assertEqual(events, ["train", "final_eval", "graph"])

    def test_final_eval_marks_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, patch("train.OptimalSolver.optimal_moves", return_value={0, 1, 2, 3, 4, 5, 6}):
            path = Path(directory) / "best_model.pt"
            checkpoints = Path(directory) / "checkpoints"
            checkpoints.mkdir()
            torch.save(self.best.state_dict(), path)
            torch.save(self.best.state_dict(), checkpoints / "checkpoint_300.pt")
            with torch.no_grad():
                next(self.model.parameters()).add_(1)
            torch.save(self.model.state_dict(), checkpoints / "checkpoint_40.pt")

            final_eval(path, 2, num_simulations=2, make_model=lambda: AlphaC4Zero(num_blocks=1, filters=4))
            with (Path(directory) / "final_evaluation.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([int(row["training_moves"]) for row in rows], [40, 300])
            self.assertEqual([row["is_best"] for row in rows], ["False", "True"])
            self.assertEqual([float(row["optimal_moves_percent"]) for row in rows], [100, 100])
            for row in rows:
                for prefix in ("", "optimal_"):
                    wins, draws, losses = (int(row[f"{prefix}{key}"]) for key in ("wins", "draws", "losses"))
                    self.assertEqual(wins + draws + losses, 2)
                    self.assertAlmostEqual(float(row[f"{prefix}match_score"]), 100 * (wins + 0.5 * draws) / 2)
                    self.assertAlmostEqual(float(row[f"{prefix}mean_result"]), (wins - losses) / 2)
            for name in ("final_match_score_vs_best.png", "final_match_score_optimal.png",
                         "final_mean_result_vs_best.png", "final_mean_result_optimal.png"):
                self.assertGreater((Path(directory) / name).stat().st_size, 0)
            self.assertFalse((Path(directory) / "optimal_moves.png").exists())

    def test_final_eval_without_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            final_eval(Path(directory) / "best_model.pt", 2, num_simulations=2)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_graph_with_no_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            graph_results(directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_training_passes_move_count_without_history(self):
        with tempfile.TemporaryDirectory() as directory, patch("train.eval") as evaluate:
            train(self.model, self.best, Path(directory) / "best_model.pt", 2, 1, 2, "cpu", 2, 2, 1000, .001)
            saved = sorted(int(path.stem.split("_")[1]) for path in (Path(directory) / "checkpoints").glob("checkpoint_*.pt"))
        self.assertEqual(evaluate.call_count, 2)
        first, second = [call.kwargs for call in evaluate.call_args_list]
        self.assertGreater(first["total_training_moves"], 0)
        self.assertGreater(second["total_training_moves"], first["total_training_moves"])
        self.assertEqual(saved, [first["total_training_moves"], second["total_training_moves"]])
        self.assertNotIn("history", first)
        self.assertNotIn("benchmark", first)


if __name__ == "__main__":
    unittest.main()
