'''**I did not write this code, credit goes to Claude Opus 5.5**'''

import argparse
import os
import sys
import tkinter as tk
from pathlib import Path

import torch

if __package__:
    from .model import AlphaC4Zero
    from .board import Board
    from .mcts import MCTS
else:
    from model import AlphaC4Zero
    from board import Board
    from mcts import MCTS

START_CELL = 72 # starting size of one board cell, it scales with the window after that
PAD = 14 # margin around the board
HOVER = 0.78 # height of the strip above the board that previews your stone, in cells
BAR_HEIGHT = 60 # tallest stats bar (100%)
FONT = "Helvetica"
THEME = {
    "bg": "#f4f5f7",
    "board": "#1d4ed8",
    "hole": "#e8edf7",
    "human": "#dc2626", "human_edge": "#991b1b", "human_ghost": "#f5b5b5",
    "model": "#facc15", "model_edge": "#ca8a04",
    "win": "#16a34a",
    "text": "#111827", "muted": "#6b7280", "track": "#e5e7eb", "rule": "#d1d5db",
    "prior": "#93c5fd", "prior_text": "#3b82f6",
    "search": "#1e3a8a",
    "good": "#16a34a", "bad": "#dc2626",
}


def winning_cells(rows, player):
    '''Cells of the first four-in-a-row for player, or [] if there is none'''
    for row in range(6):
        for col in range(7):
            for d_row, d_col in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(row + i * d_row, col + i * d_col) for i in range(4)]
                if all(0 <= r < 6 and 0 <= c < 7 and rows[r][c] == player for r, c in cells):
                    return cells
    return []


class Game:
    def __init__(self, root, model, num_simulations):
        self.root = root
        self.model = model
        self.num_simulations = num_simulations
        self.hover_col = None
        width = 7 * START_CELL + 2 * PAD
        root.configure(bg=THEME["bg"])

        self.status = tk.Label(root, font=(FONT, 18, "bold"), bg=THEME["bg"])
        self.status.pack(pady=(12, 0))

        # Pack the fixed-height pieces from the bottom first, so the board takes whatever space is left
        buttons = tk.Frame(root, bg=THEME["bg"])
        buttons.pack(side="bottom", pady=(8, 16))
        tk.Button(buttons, text="New game: you first", command=lambda: self.new_game(human=1)).pack(side="left", padx=6)
        tk.Button(buttons, text="New game: model first", command=lambda: self.new_game(human=-1)).pack(side="left", padx=6)

        # Model's evaluation: one gauge each for the network and the search, from -1 (loss) to 1 (win)
        self.eval_canvas = tk.Canvas(root, width=width, height=96, bg=THEME["bg"], highlightthickness=0)
        self.eval_canvas.pack(side="bottom", fill="x", padx=12, pady=(10, 0))

        legend = tk.Frame(root, bg=THEME["bg"])
        legend.pack(side="bottom")
        for color, text in ((THEME["prior"], "network prior %"), (THEME["search"], "search visits % (most visited is played)")):
            swatch = tk.Canvas(legend, width=12, height=12, bg=THEME["bg"], highlightthickness=0)
            swatch.create_rectangle(1, 1, 11, 11, fill=color, outline="")
            swatch.pack(side="left", padx=(10, 4))
            tk.Label(legend, text=text, font=(FONT, 12), fg=THEME["muted"], bg=THEME["bg"]).pack(side="left")

        # Model's move stats: one pair of bars per column, lined up with the board
        self.stats_canvas = tk.Canvas(root, width=width, height=BAR_HEIGHT + 44, bg=THEME["bg"], highlightthickness=0)
        self.stats_canvas.pack(side="bottom", fill="x", padx=12)

        self.canvas = tk.Canvas(root, width=width, height=(6 + HOVER) * START_CELL + 2 * PAD, bg=THEME["bg"], highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=12)
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Motion>", self.hover)
        self.canvas.bind("<Leave>", lambda event: self.set_hover(None))
        for canvas in (self.canvas, self.stats_canvas, self.eval_canvas):
            canvas.bind("<Configure>", lambda event: self.draw())
        for col in range(7):
            root.bind(str(col + 1), lambda event, col=col: self.human_move(col))

        self.new_game(human=1)

    def geometry(self):
        '''Cell size and board corner for the current board canvas size, the panels below use the same columns'''
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width <= 1: # not laid out yet
            width, height = float(self.canvas["width"]), float(self.canvas["height"])
        cell = max(20, min((width - 2 * PAD) / 7, (height - 2 * PAD) / (6 + HOVER)))
        left = (width - 7 * cell - 2 * PAD) / 2
        top = (height - (6 + HOVER) * cell - 2 * PAD) / 2
        return cell, left, top

    def new_game(self, human):
        self.board = Board()
        self.to_play = 1
        self.human = human
        self.over = False
        self.stats = None
        self.values = None
        self.last_move = None
        self.win = []
        self.draw()
        if self.to_play == self.human:
            self.set_status("Your move", THEME["human"])
        else:
            self.think()

    def set_status(self, text, color=THEME["text"]):
        self.status.config(text=text, fg=color)

    def column_at(self, x):
        cell, left, _ = self.geometry()
        col = int((x - left - PAD) // cell)
        return col if 0 <= col < 7 else None

    def human_turn(self):
        return not self.over and self.to_play == self.human

    def hover(self, event):
        self.set_hover(self.column_at(event.x))

    def set_hover(self, col):
        if col != self.hover_col:
            self.hover_col = col
            self.draw_board()

    def click(self, event):
        col = self.column_at(event.x)
        if col is not None:
            self.human_move(col)

    def human_move(self, col):
        if not self.human_turn() or not self.board.get_moves()[col]:
            return
        self.play(col)
        if not self.over:
            self.think()

    def think(self):
        self.set_status("Model is thinking...", THEME["muted"])
        # let the window redraw before the search blocks it
        self.root.after(50, self.model_move)

    def model_move(self):
        # a new game may have started while this move was scheduled
        if self.over or self.to_play == self.human:
            return
        tree = MCTS(self.board, self.model, to_play=self.to_play)
        move = tree.search(self.num_simulations)
        node = tree.root_node

        # Network outputs for the position the model moved from, same encoding as MCTS.expand
        pieces = self.board.pieces
        planes = torch.stack((pieces == -self.to_play, pieces == self.to_play), dim=-1).unsqueeze(0)
        with torch.no_grad():
            probs, value = self.model(planes)

        total_visits = sum(node.visit_counts.values())
        self.stats = {
            "move": move,
            "prior": probs[0].tolist(),
            "search": [node.visit_counts.get(col, 0) / total_visits for col in range(7)],
        }
        self.values = {"network": value.item(), "search": node.accum_results[move] / node.visit_counts[move]}

        self.play(move)
        if not self.over:
            self.set_status(f"Model played column {move + 1}. Your move", THEME["human"])

    def play(self, col):
        player = self.to_play
        self.board.add_stone(col, player)
        self.last_move = (int((self.board.pieces[:, col] != 0).nonzero()[0]), col)
        self.to_play = -player
        if self.board.has_won(player):
            self.over = True
            self.win = winning_cells(self.board.pieces.tolist(), player)
            self.set_status("You win!" if player == self.human else "Model wins!", THEME["win"])
        elif self.board.full():
            self.over = True
            self.set_status("Draw!")
        self.draw()

    def draw(self):
        self.draw_board()
        self.draw_stats()
        self.draw_evaluation()

    def draw_board(self):
        c = self.canvas
        c.delete("all")
        cell, left, top = self.geometry()
        rows = self.board.pieces.tolist()

        # Preview of your stone above the hovered column
        if self.hover_col is not None and self.human_turn() and self.board.get_moves()[self.hover_col]:
            x, y = left + PAD + (self.hover_col + 0.5) * cell, top + HOVER * cell / 2
            r = 0.36 * cell
            c.create_oval(x - r, y - r, x + r, y + r, fill=THEME["human_ghost"], outline="")

        board_top = top + HOVER * cell
        c.create_rectangle(left, board_top, left + 7 * cell + 2 * PAD, board_top + 6 * cell + 2 * PAD, fill=THEME["board"], outline="")
        inset, dot = 0.1 * cell, 0.3 * cell # dot = inset from the stone edge for the last-move marker
        for row in range(6):
            for col in range(7):
                x1, y1 = left + PAD + col * cell + inset, board_top + PAD + row * cell + inset
                x2, y2 = x1 + cell - 2 * inset, y1 + cell - 2 * inset
                value = rows[row][col]
                if value == 0:
                    c.create_oval(x1, y1, x2, y2, fill=THEME["hole"], outline="")
                    continue
                who = "human" if value == self.human else "model"
                c.create_oval(x1, y1, x2, y2, fill=THEME[who], outline=THEME[f"{who}_edge"], width=3)
                if (row, col) in self.win:
                    c.create_oval(x1 - 3, y1 - 3, x2 + 3, y2 + 3, outline=THEME["win"], width=5)
                elif (row, col) == self.last_move:
                    c.create_oval(x1 + dot, y1 + dot, x2 - dot, y2 - dot, fill=THEME[f"{who}_edge"], outline="")

    def draw_stats(self):
        c = self.stats_canvas
        c.delete("all")
        cell, left, _ = self.geometry()
        base = 18 + BAR_HEIGHT # bars grow up from here, labels above them, column numbers below
        c.create_line(left + PAD, base, left + PAD + 7 * cell, base, fill=THEME["rule"])
        bar, offset = 0.3 * cell, 0.18 * cell
        for col in range(7):
            x = left + PAD + (col + 0.5) * cell
            chosen = self.stats is not None and col == self.stats["move"]
            if chosen:
                c.create_oval(x - 11, base + 4, x + 11, base + 26, fill=THEME["model"], outline="")
            c.create_text(x, base + 15, text=str(col + 1), font=(FONT, 12, "bold"), fill=THEME["text"] if chosen else THEME["muted"])
            if self.stats is None:
                continue
            for key, shift in (("prior", -offset), ("search", offset)):
                share = self.stats[key][col]
                if share > 0:
                    c.create_rectangle(x + shift - bar / 2, base - max(2, share * BAR_HEIGHT), x + shift + bar / 2, base, fill=THEME[key], outline="")
                c.create_text(x + shift, base - share * BAR_HEIGHT - 9, text=f"{100 * share:.0f}",
                              font=(FONT, 11, "bold" if chosen else "normal"),
                              fill=THEME["prior_text"] if key == "prior" else THEME["search"])
        if self.stats is None:
            c.create_text(left + PAD + 3.5 * cell, base - BAR_HEIGHT / 2, text="The model's move stats appear here after it plays",
                          font=(FONT, 12), fill=THEME["muted"])

    def draw_evaluation(self):
        c = self.eval_canvas
        c.delete("all")
        cell, left, _ = self.geometry()
        x1, x2 = left + PAD, left + PAD + 7 * cell # same width as the board

        if self.values is None:
            title = "Model's evaluation"
        elif self.values["search"] > 0.3:
            title = "Model's evaluation: it thinks it's ahead"
        elif self.values["search"] < -0.3:
            title = "Model's evaluation: it thinks it's behind"
        else:
            title = "Model's evaluation: roughly even"
        c.create_text(x1, 10, text=title, anchor="w", font=(FONT, 13, "bold"), fill=THEME["text"])

        # Gauges: track from -1 (left) to 1 (right), filled from the middle towards the value
        track_left, track_right = x1 + 70, x2 - 50
        middle = (track_left + track_right) / 2
        for i, key in enumerate(("network", "search")):
            y = 36 + 22 * i
            c.create_text(x1, y, text=key, anchor="w", font=(FONT, 12), fill=THEME["muted"])
            c.create_rectangle(track_left, y - 6, track_right, y + 6, fill=THEME["track"], outline="")
            if self.values is None:
                c.create_text(x2, y, text="-", anchor="e", font=(FONT, 12), fill=THEME["muted"])
            else:
                value = self.values[key]
                end = middle + value * (track_right - middle)
                color = THEME["good"] if value >= 0 else THEME["bad"]
                c.create_rectangle(min(middle, end), y - 6, max(middle, end), y + 6, fill=color, outline="")
                c.create_text(x2, y, text=f"{value:+.2f}", anchor="e", font=(FONT, 12, "bold"), fill=color)
            c.create_line(middle, y - 9, middle, y + 9, fill=THEME["muted"])
        for x, text, anchor in ((track_left, "loss", "w"), (middle, "draw", "center"), (track_right, "win", "e")):
            c.create_text(x, 84, text=text, anchor=anchor, font=(FONT, 11), fill=THEME["muted"])


def find_tcl_tk():
    '''uv's standalone Python can't always find its Tcl/Tk scripts from inside a venv, so point it at them'''
    lib = Path(sys.base_prefix) / "lib"
    for var, name in (("TCL_LIBRARY", "tcl"), ("TK_LIBRARY", "tk")):
        found = sorted(lib.glob(f"{name}[0-9].[0-9]*"))
        if var not in os.environ and found:
            os.environ[var] = str(found[-1])


def main():
    parser = argparse.ArgumentParser(description="Play Connect Four against a trained model.")
    parser.add_argument("--model", default=Path(__file__).resolve().parents[1] / "results" / "best_model.pt")
    parser.add_argument("--num-simulations", type=int, default=200)
    args = parser.parse_args()
    if args.num_simulations < 2:
        parser.error("num_simulations must be at least 2")

    model = AlphaC4Zero()
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()

    find_tcl_tk()
    root = tk.Tk()
    root.title("Connect 4 vs AlphaC4Zero")
    root.minsize(480, 710) # below this the stats labels start to overlap
    Game(root, model, args.num_simulations)
    root.mainloop()


if __name__ == "__main__":
    main()
