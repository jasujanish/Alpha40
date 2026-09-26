'''
Exact win/draw/loss solver for standard 6x7 Connect Four. 
This file is not written by me (as I'm not interested in writting a tree search algo). 
Credit for this file goes to GPT-6 Astra Light.
'''



class OptimalSolver:
    HEIGHT = 6
    WIDTH = 7
    ORDER = (3, 2, 4, 1, 5, 0, 6)
    BOTTOM = tuple(1 << (7 * col) for col in range(7))
    COLUMNS = tuple(63 << (7 * col) for col in range(7))

    def __init__(self, cache_size=500000):
        self.cache_size = cache_size
        self.cache = {}

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

    def _search(self, player, occupied, alpha=-2, beta=2):
        opponent = occupied ^ player
        if self.has_won(player):
            return 1
        if self.has_won(opponent):
            return -1
        moves = self.legal_moves(occupied)
        if not moves:
            return 0

        key = (player, occupied)
        original_alpha, original_beta = alpha, beta
        cached = self.cache.get(key)
        if cached is not None:
            value, bound = cached
            if bound == "exact":
                return value
            if bound == "lower":
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                return value

        # An immediate win takes precedence over having to block a threat.
        if any(self.has_won(player | bit) for _, bit in moves):
            return 1
        threats = [(col, bit) for col, bit in moves if self.has_won(opponent | bit)]
        if len(threats) > 1:
            return -1
        if threats:
            moves = threats

        best = -2
        for _, bit in moves:
            value = -self._search(opponent, occupied | bit, -beta, -alpha)
            best = max(best, value)
            alpha = max(alpha, best)
            if alpha >= beta or best == 1:
                break

        bound = ("upper" if best <= original_alpha else
                 "lower" if best >= original_beta else "exact")
        if len(self.cache) >= self.cache_size:
            self.cache.clear()
        self.cache[key] = (best, bound)
        return best

    def solve(self, board, to_play):
        """Return +1 (forced win), 0 (draw), or -1 (forced loss)."""
        return self._search(*self.encode(board, to_play))

    def move_scores(self, board, to_play):
        """Exact outcomes per legal column, from to_play's perspective."""
        player, occupied = self.encode(board, to_play)
        opponent = occupied ^ player
        if self.has_won(player) or self.has_won(opponent):
            return {}
        return {
            col: (1 if self.has_won(player | bit)
                  else -self._search(opponent, occupied | bit))
            for col, bit in self.legal_moves(occupied)
        }

    def optimal_moves(self, board, to_play):
        scores = self.move_scores(board, to_play)
        if not scores:
            return set()
        best = max(scores.values())
        return {col for col, score in scores.items() if score == best}

