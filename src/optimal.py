'''**I did not write this code, credit goes to Claude Opus 5.5**'''

import json
import subprocess
from pathlib import Path

SOLVER_DIR = Path(__file__).resolve().parents[1] / "external" / "connect4"


class OptimalSolver:
    HEIGHT = 6
    WIDTH = 7
    ORDER = (3, 2, 4, 1, 5, 0, 6)
    BOTTOM = tuple(1 << (7 * col) for col in range(7))
    COLUMNS = tuple(63 << (7 * col) for col in range(7))

    def __init__(self, cache_path=None, solver_path=SOLVER_DIR / "c4solver", book_path=SOLVER_DIR / "7x6.book"):
        self.solver_path = Path(solver_path)
        self.book_path = Path(book_path)
        self.process = None # started on the first uncached query
        self.cache = {} # (player, occupied) -> {col: -1/0/1}
        self.cache_file = None
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                with cache_path.open() as file:
                    for line in file:
                        entry = json.loads(line)
                        self.cache[(entry["player"], entry["occupied"])] = {int(col): score for col, score in entry["scores"].items()}
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file = cache_path.open("a")

    @staticmethod
    def has_won(pieces):
        for shift in (1, 7, 6, 8):
            pairs = pieces & (pieces >> shift)
            if pairs & (pairs >> (2 * shift)):
                return True
        return False

    def encode(self, board, to_play):
        if (board.height, board.width, board.win_length) != (6, 7, 4):
            raise ValueError("OptimalSolver requires a 6x7 board with win_length=4")
        if to_play not in (-1, 1):
            raise ValueError("to_play must be -1 or 1")
        rows = board.pieces.detach().cpu().tolist()
        player, occupied = 0, 0
        for col in range(7):
            found_empty = False
            for row in range(5, -1, -1):
                value = rows[row][col]
                if value not in (-1, 0, 1):
                    raise ValueError("Board cells must be -1, 0, or 1")
                if value == 0:
                    found_empty = True
                    continue
                if found_empty:
                    raise ValueError("Board violates gravity")
                bit = 1 << (7 * col + 5 - row)
                occupied |= bit
                if value == to_play:
                    player |= bit
        return player, occupied

    def legal_moves(self, occupied):
        return [(col, (occupied + self.BOTTOM[col]) & self.COLUMNS[col])
                for col in self.ORDER
                if (occupied + self.BOTTOM[col]) & self.COLUMNS[col]]

    def move_sequence(self, player, occupied):
        '''Find an alternating, bottom-up move order (as column indices) that reaches this position'''
        opponent = occupied ^ player
        player_stones, opponent_stones = bin(player).count("1"), bin(opponent).count("1")
        if player_stones == opponent_stones:
            first = player
        elif opponent_stones == player_stones + 1:
            first = opponent
        else:
            raise ValueError("Stone counts are inconsistent with to_play")

        # Owner of each stone per column, bottom up (True == first mover)
        columns = []
        for col in range(7):
            stones = []
            for row in range(6):
                bit = 1 << (7 * col + row)
                if not occupied & bit:
                    break
                stones.append(bool(first & bit))
            columns.append(stones)

        # DFS over column heights, the side to move is fixed by how many stones are down
        total = bin(occupied).count("1")
        dead_ends = set()
        def search(heights, sequence):
            if len(sequence) == total:
                return True
            if heights in dead_ends:
                return False
            first_to_move = len(sequence) % 2 == 0
            for col in range(7):
                height = heights[col]
                if height < len(columns[col]) and columns[col][height] == first_to_move:
                    sequence.append(col)
                    if search(heights[:col] + (height + 1,) + heights[col + 1:], sequence):
                        return True
                    sequence.pop()
            dead_ends.add(heights)
            return False

        sequence = []
        if not search((0,) * 7, sequence):
            raise ValueError("Position is not reachable by alternating play")
        return sequence

    def query(self, sequence):
        '''Ask the Pons solver for the weak (win/draw/loss) score of every column'''
        if self.process is None:
            for path in (self.solver_path, self.book_path):
                if not path.exists():
                    raise FileNotFoundError(f"{path} not found, see the README to build the Connect Four solver")
            self.process = subprocess.Popen(
                [str(self.solver_path), "-w", "-a", "-b", str(self.book_path)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
        # Pons columns are 1-indexed. Invalid positions print nothing, so they must be filtered out before this point
        self.process.stdin.write("".join(str(col + 1) for col in sequence) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Connect Four solver exited unexpectedly")
        scores = [int(score) for score in line.split()[-7:]]
        # -1000 marks a full column, otherwise only the sign matters
        return {col: (score > 0) - (score < 0) for col, score in enumerate(scores) if score != -1000}

    def solve(self, board, to_play):
        """Return +1 (forced win), 0 (draw), or -1 (forced loss)."""
        player, occupied = self.encode(board, to_play)
        opponent = occupied ^ player
        if self.has_won(player):
            return 1
        if self.has_won(opponent):
            return -1
        # Immediate wins don't need the solver (or a reachable position)
        if any(self.has_won(player | bit) for _, bit in self.legal_moves(occupied)):
            return 1
        scores = self.move_scores(board, to_play)
        return max(scores.values()) if scores else 0

    def move_scores(self, board, to_play):
        """Exact outcomes per legal column, from to_play's perspective."""
        player, occupied = self.encode(board, to_play)
        opponent = occupied ^ player
        if self.has_won(player) or self.has_won(opponent) or not self.legal_moves(occupied):
            return {}

        key = (player, occupied)
        if key not in self.cache:
            scores = self.query(self.move_sequence(player, occupied))
            self.cache[key] = scores
            if self.cache_file is not None:
                self.cache_file.write(json.dumps({"player": player, "occupied": occupied, "scores": scores}) + "\n")
                self.cache_file.flush()
        return dict(self.cache[key])

    def optimal_moves(self, board, to_play):
        scores = self.move_scores(board, to_play)
        if not scores:
            return set()
        best = max(scores.values())
        return {col for col, score in scores.items() if score == best}

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            self.process.wait()
            self.process = None
        if self.cache_file is not None:
            self.cache_file.close()
            self.cache_file = None

    def __del__(self):
        self.close()
