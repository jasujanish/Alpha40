'''**I did not write this code, credit goes to Claude Opus 5.5**'''

import argparse
import os
import sys
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path

import torch

if __package__:
    from .model import AlphaC4Zero
    from .board import Board
    from .mcts import MCTS
    from .optimal import OptimalSolver
else:
    from model import AlphaC4Zero
    from board import Board
    from mcts import MCTS
    from optimal import OptimalSolver

START_CELL = 66 # starting size of one board cell, it scales with the window after that
PAD = 14 # margin between the board edge and the holes
GAP = 20 # space around and between the main areas
SIDEBAR = 360 # sidebar width
CARD_GAP = 12 # space between sidebar cards
CARDS = {"move": 232, "evaluation": 132, "optimality": 176} # card heights
COLORS = {
    "bg": "#f5f5f7", "card": "#ffffff", "card_edge": "#e5e5ea",
    "text": "#1d1d1f", "secondary": "#86868b", "track": "#e8e8ed",
    "accent": "#0071e3", "accent_hover": "#0077ed",
    "board": "#1f5fcc", "board_hover": "#2d6ad6", "hole": "#f5f5f7",
    "human": "#ff3b30", "human_edge": "#d70015", "human_ghost": "#ffb4ae",
    "model": "#ffcc00", "model_edge": "#e0a800",
    "network": "#a8c7fa", "search": "#0071e3",
    "good": "#34c759", "bad": "#ff3b30", "bad_text": "#d70015",
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


def rounded_rect(canvas, x1, y1, x2, y2, radius, **options):
    '''Rounded rectangle as a smoothed polygon, Tk has no native one (doubled points keep the edges straight)'''
    r = max(0, min(radius, (x2 - x1) / 2, (y2 - y1) / 2))
    points = (x1 + r, y1, x1 + r, y1, x2 - r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y1 + r, x2, y2 - r, x2, y2 - r, x2, y2,
              x2 - r, y2, x2 - r, y2, x1 + r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y2 - r, x1, y1 + r, x1, y1 + r, x1, y1)
    return canvas.create_polygon(points, smooth=True, splinesteps=16, **options)


class Game:
    def __init__(self, root, model, num_simulations):
        self.root = root
        self.model = model
        self.num_simulations = num_simulations
        self.family = tkfont.nametofont("TkDefaultFont").actual("family") # the system font (SF on macOS)
        self.solver = OptimalSolver() # grades every move, the process starts on the first query
        self.hover_col = None
        self.hover_button = None
        self.button_items = {} # header canvas items recoloured on hover
        self.first = 1 # who starts the next game, set by the segmented control
        root.configure(bg=COLORS["bg"])
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        # Header: status on the left, who-starts control and New Game on the right
        self.header = tk.Canvas(root, height=44, width=7 * START_CELL + 2 * PAD + GAP + SIDEBAR, bg=COLORS["bg"], highlightthickness=0)
        self.header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=GAP, pady=(GAP - 4, 8))
        for tag, action in (("new_game", lambda: self.new_game(self.first)), ("seg_human", lambda: self.choose_first(1)), ("seg_model", lambda: self.choose_first(-1))):
            self.header.tag_bind(tag, "<Button-1>", lambda event, action=action: action())
            self.header.tag_bind(tag, "<Enter>", lambda event, tag=tag: self.set_button_hover(tag))
            self.header.tag_bind(tag, "<Leave>", lambda event: self.set_button_hover(None))

        self.canvas = tk.Canvas(root, width=7 * START_CELL + 2 * PAD, height=6 * START_CELL + 2 * PAD, bg=COLORS["bg"], highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=(GAP, 0), pady=(0, GAP))
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Motion>", self.hover)
        self.canvas.bind("<Leave>", lambda event: self.set_hover(None))

        self.sidebar = tk.Canvas(root, width=SIDEBAR, height=sum(CARDS.values()) + CARD_GAP * (len(CARDS) - 1), bg=COLORS["bg"], highlightthickness=0)
        self.sidebar.grid(row=1, column=1, sticky="n", padx=GAP, pady=(0, GAP))

        for canvas in (self.header, self.canvas, self.sidebar):
            canvas.bind("<Configure>", lambda event: self.draw())
        for col in range(7):
            root.bind(str(col + 1), lambda event, col=col: self.human_move(col))

        self.new_game(self.first)

    def font(self, size, bold=False):
        return (self.family, size, "bold" if bold else "normal")

    def text_width(self, text, size):
        return tkfont.Font(family=self.family, size=size).measure(text)

    def geometry(self):
        '''Cell size and board corner for the current board canvas size'''
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width <= 1: # not laid out yet
            width, height = float(self.canvas["width"]), float(self.canvas["height"])
        cell = max(20, min((width - 2 * PAD) / 7, (height - 2 * PAD) / 6))
        left = (width - 7 * cell - 2 * PAD) / 2
        return cell, left, 0 # top aligned, so the board lines up with the sidebar

    # Game flow

    def new_game(self, human):
        self.board = Board()
        self.to_play = 1
        self.human = human
        self.over = False
        self.moves_played = 0
        self.stats = None
        self.values = None
        self.last_move = None
        self.win = []
        # per side: moves graded, optimal moves, (best, played) outcome of the last move
        self.quality = {who: {"moves": 0, "optimal": 0, "last": None} for who in ("human", "model")}
        self.outlook = None # perfect-play result from the current position, from your view
        if self.to_play == self.human:
            self.set_status("Your turn", "human")
        else:
            self.think()

    def choose_first(self, human):
        self.first = human
        # switching sides before anyone has moved just restarts, otherwise it applies to the next game
        if self.moves_played == 0 or self.over:
            self.new_game(human)
        else:
            self.draw_header()

    def set_status(self, text, dot):
        self.status = (text, dot)
        self.draw()

    def human_turn(self):
        return not self.over and self.to_play == self.human

    def human_move(self, col):
        if not self.human_turn() or not self.board.get_moves()[col]:
            return
        self.play(col)
        if not self.over:
            self.think()

    def think(self):
        self.set_status("Model is thinking…", "model")
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
            "network": probs[0].tolist(),
            "search": [node.visit_counts.get(col, 0) / total_visits for col in range(7)],
        }
        self.values = {"network": value.item(), "search": node.accum_results[move] / node.visit_counts[move]}

        self.play(move)
        if not self.over:
            self.set_status("Your turn", "human")

    def grade(self, col):
        '''Ask the solver whether playing col keeps the best result available (win > draw > loss)'''
        if self.solver is None:
            return
        try:
            scores = self.solver.move_scores(self.board, self.to_play)
        except (OSError, RuntimeError, ValueError):
            self.solver = None # not built (see scripts/setup.sh) or failed, stop grading
            return
        best, played = max(scores.values()), scores[col]
        side = self.quality["human" if self.to_play == self.human else "model"]
        side["moves"] += 1
        side["optimal"] += played == best
        side["last"] = (best, played)
        self.outlook = played if self.to_play == self.human else -played

    def play(self, col):
        self.grade(col)
        player = self.to_play
        self.board.add_stone(col, player)
        self.moves_played += 1
        self.last_move = (int((self.board.pieces[:, col] != 0).nonzero()[0]), col)
        self.to_play = -player
        if self.board.has_won(player):
            self.over = True
            self.win = winning_cells(self.board.pieces.tolist(), player)
            self.set_status("You won" if player == self.human else "The model won", "human" if player == self.human else "model")
        elif self.board.full():
            self.over = True
            self.set_status("Draw", None)
        else:
            self.draw()

    # Input

    def column_at(self, x):
        cell, left, _ = self.geometry()
        col = int((x - left - PAD) // cell)
        return col if 0 <= col < 7 else None

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

    def set_button_hover(self, tag):
        # only recolour here, redrawing the header deletes the item under the pointer, which fires
        # Leave/Enter again and would redraw forever
        if tag == self.hover_button:
            return
        self.hover_button = tag
        self.header.config(cursor="pointinghand" if tag else "")
        self.apply_button_hover()

    def apply_button_hover(self):
        c = self.header
        if "new_game" in self.button_items:
            c.itemconfigure(self.button_items["new_game"], fill=COLORS["accent_hover"] if self.hover_button == "new_game" else COLORS["accent"])
        for tag, human in (("seg_human", 1), ("seg_model", -1)):
            if tag in self.button_items:
                selected = self.first == human or self.hover_button == tag
                c.itemconfigure(self.button_items[tag], fill=COLORS["text"] if selected else COLORS["secondary"])

    # Drawing

    def draw(self):
        self.draw_header()
        self.draw_board()
        self.draw_sidebar()

    def draw_header(self):
        c = self.header
        c.delete("all")
        width = c.winfo_width() if c.winfo_width() > 1 else float(c["width"])

        text, dot = self.status
        c.create_oval(1, 17, 11, 27, fill=COLORS[dot] if dot else COLORS["secondary"], outline="")
        c.create_text(20, 22, text=text, anchor="w", font=self.font(15), fill=COLORS["text"])

        # New Game pill button
        x2, y1, y2 = width, 7, 37
        x1 = x2 - 108
        self.button_items["new_game"] = rounded_rect(c, x1, y1, x2, y2, 15, fill=COLORS["accent"], outline="", tags="new_game")
        c.create_text((x1 + x2) / 2, (y1 + y2) / 2, text="New Game", font=self.font(13, bold=True), fill="white", tags="new_game")

        # Segmented control for who moves first
        seg_right = x1 - 12
        seg_left = seg_right - 200
        rounded_rect(c, seg_left, y1, seg_right, y2, 9, fill=COLORS["track"], outline="")
        half = (seg_right - seg_left) / 2
        for i, (tag, label, human) in enumerate((("seg_human", "You first", 1), ("seg_model", "Model first", -1))):
            sx1, sx2 = seg_left + i * half, seg_left + (i + 1) * half
            if self.first == human:
                rounded_rect(c, sx1 + 2, y1 + 2, sx2 - 2, y2 - 2, 7, fill=COLORS["card"], outline=COLORS["card_edge"], tags=tag)
            else:
                # invisible hit area so the whole segment is clickable
                rounded_rect(c, sx1 + 2, y1 + 2, sx2 - 2, y2 - 2, 7, fill=COLORS["track"], outline="", tags=tag)
            self.button_items[tag] = c.create_text((sx1 + sx2) / 2, (y1 + y2) / 2, text=label, font=self.font(12), tags=tag)
        self.apply_button_hover()

    def draw_board(self):
        c = self.canvas
        c.delete("all")
        cell, left, top = self.geometry()
        rows = self.board.pieces.tolist()

        board_top, board_bottom = top, top + 6 * cell + 2 * PAD
        rounded_rect(c, left, board_top, left + 7 * cell + 2 * PAD, board_bottom, 18, fill=COLORS["board"], outline="")

        # Hovered column: a lighter column, and a faded stone in the hole it would land in
        preview = None
        if self.hover_col is not None and self.human_turn() and self.board.get_moves()[self.hover_col]:
            x1 = left + PAD + self.hover_col * cell
            rounded_rect(c, x1 + 2, board_top + PAD - 4, x1 + cell - 2, board_bottom - PAD + 4, cell / 2, fill=COLORS["board_hover"], outline="")
            preview = (max(row for row in range(6) if rows[row][self.hover_col] == 0), self.hover_col)

        inset, dot = 0.09 * cell, 0.32 * cell
        for row in range(6):
            for col in range(7):
                x1, y1 = left + PAD + col * cell + inset, board_top + PAD + row * cell + inset
                x2, y2 = x1 + cell - 2 * inset, y1 + cell - 2 * inset
                value = rows[row][col]
                if value == 0:
                    c.create_oval(x1, y1, x2, y2, fill=COLORS["human_ghost"] if (row, col) == preview else COLORS["hole"], outline="")
                    continue
                who = "human" if value == self.human else "model"
                c.create_oval(x1, y1, x2, y2, fill=COLORS[who], outline=COLORS[f"{who}_edge"], width=2)
                if (row, col) in self.win:
                    c.create_oval(x1 - 3, y1 - 3, x2 + 3, y2 + 3, outline="white", width=4)
                elif (row, col) == self.last_move:
                    c.create_oval(x1 + dot, y1 + dot, x2 - dot, y2 - dot, fill=COLORS[f"{who}_edge"], outline="")

    def card(self, top, height, title, detail=""):
        '''White rounded card with a title and an optional grey detail on the right'''
        c = self.sidebar
        rounded_rect(c, 1, top + 1, SIDEBAR - 1, top + height - 1, 12, fill=COLORS["card"], outline=COLORS["card_edge"])
        c.create_text(16, top + 22, text=title, anchor="w", font=self.font(14, bold=True), fill=COLORS["text"])
        if detail:
            c.create_text(SIDEBAR - 16, top + 22, text=detail, anchor="e", font=self.font(12), fill=COLORS["secondary"])

    def draw_sidebar(self):
        self.sidebar.delete("all")
        top = 0
        for name, draw in (("move", self.draw_move_card), ("evaluation", self.draw_evaluation_card), ("optimality", self.draw_optimality_card)):
            draw(top)
            top += CARDS[name] + CARD_GAP

    def draw_move_card(self, top):
        '''Network prior and MCTS visit share for each column, for the model's last move'''
        c = self.sidebar
        chosen = self.stats["move"] if self.stats else None
        self.card(top, CARDS["move"], "Model's move")

        net_x, search_x, bar_left, bar_right = SIDEBAR - 104, SIDEBAR - 16, 48, SIDEBAR - 176
        header_y = top + 52
        for x, label, key in ((net_x, "Network", "network"), (search_x, "Search", "search")):
            c.create_text(x, header_y, text=label, anchor="e", font=self.font(11), fill=COLORS["secondary"])
            key_x = x - self.text_width(label, 11) - 6
            c.create_oval(key_x - 8, header_y - 4, key_x, header_y + 4, fill=COLORS[key], outline="")

        for col in range(7):
            y = top + 80 + 22 * col
            is_chosen = col == chosen
            if is_chosen:
                c.create_oval(14, y - 8, 30, y + 8, fill=COLORS["model"], outline="")
            c.create_text(22, y, text=str(col + 1), font=self.font(12, bold=is_chosen), fill=COLORS["text"] if is_chosen else COLORS["secondary"])
            if not self.stats:
                c.create_text(search_x, y, text="–", anchor="e", font=self.font(12), fill=COLORS["secondary"])
                continue
            network, search = self.stats["network"][col], self.stats["search"][col]
            span = bar_right - bar_left
            if network > 0:
                rounded_rect(c, bar_left, y - 6, bar_left + max(5, network * span), y - 1, 2, fill=COLORS["network"], outline="")
            if search > 0:
                rounded_rect(c, bar_left, y + 1, bar_left + max(5, search * span), y + 6, 2, fill=COLORS["search"], outline="")
            c.create_text(net_x, y, text=f"{100 * network:.0f}%", anchor="e", font=self.font(12), fill=COLORS["secondary"])
            c.create_text(search_x, y, text=f"{100 * search:.0f}%", anchor="e", font=self.font(12, bold=is_chosen), fill=COLORS["text"])

    def draw_evaluation_card(self, top):
        '''Network and search value of the position the model moved from, from its view'''
        c = self.sidebar
        self.card(top, CARDS["evaluation"], "Model evaluation")

        track_left, track_right = 84, SIDEBAR - 64
        middle = (track_left + track_right) / 2
        for i, key in enumerate(("network", "search")):
            y = top + 50 + 22 * i
            c.create_text(16, y, text=key.capitalize(), anchor="w", font=self.font(12), fill=COLORS["secondary"])
            rounded_rect(c, track_left, y - 3, track_right, y + 3, 3, fill=COLORS["track"], outline="")
            if self.values is None:
                c.create_text(SIDEBAR - 16, y, text="–", anchor="e", font=self.font(12), fill=COLORS["secondary"])
            else:
                value = self.values[key]
                end = middle + value * (track_right - middle)
                if abs(end - middle) >= 1:
                    rounded_rect(c, min(middle, end), y - 3, max(middle, end), y + 3, 3, fill=COLORS["good"] if value >= 0 else COLORS["bad"], outline="")
                c.create_text(SIDEBAR - 16, y, text=f"{value:+.2f}", anchor="e", font=self.font(12), fill=COLORS["text"])
            c.create_line(middle, y - 7, middle, y + 7, fill=COLORS["secondary"])
        for x, text, anchor in ((track_left, "Loss", "w"), (middle, "Draw", "center"), (track_right, "Win", "e")):
            c.create_text(x, top + 90, text=text, anchor=anchor, font=self.font(10), fill=COLORS["secondary"])
        if self.values is not None:
            search = self.values["search"]
            summary = "Model thinks it's ahead" if search > 0.3 else "Model thinks it's behind" if search < -0.3 else "Roughly even"
            c.create_text(16, top + 112, text=summary, anchor="w", font=self.font(12), fill=COLORS["text"])

    def draw_optimality_card(self, top):
        '''How often you and the model played a solver-optimal move, and who wins with perfect play from here'''
        c = self.sidebar
        self.card(top, CARDS["optimality"], "Optimality")
        if self.solver is None:
            c.create_text(SIDEBAR / 2, top + 90, text="Build the solver with scripts/setup.sh\nto grade moves", justify="center",
                          font=self.font(12), fill=COLORS["secondary"])
            return

        outcome = {1: "win", 0: "draw", -1: "loss"}
        for i, (who, name) in enumerate((("human", "You"), ("model", "Model"))):
            y = top + 52 + 50 * i
            side = self.quality[who]
            c.create_oval(16, y - 6, 28, y + 6, fill=COLORS[who], outline="")
            c.create_text(36, y, text=name, anchor="w", font=self.font(13), fill=COLORS["text"])
            count = f"{side['optimal']} / {side['moves']}" if side["moves"] else "–"
            c.create_text(SIDEBAR - 16, y, text=count, anchor="e", font=self.font(12), fill=COLORS["text"] if side["moves"] else COLORS["secondary"])
            rounded_rect(c, 16, y + 13, SIDEBAR - 16, y + 18, 3, fill=COLORS["track"], outline="")
            if side["moves"]:
                share = side["optimal"] / side["moves"]
                if share > 0:
                    rounded_rect(c, 16, y + 13, 16 + max(5, share * (SIDEBAR - 32)), y + 18, 3, fill=COLORS["good"], outline="")
                best, played = side["last"]
                if played != best: # only flag bad moves
                    kind = "Blunder" if best - played == 2 else "Mistake" # win -> loss is a blunder
                    c.create_text(16, y + 31, text=f"{kind}: {outcome[best]} → {outcome[played]}", anchor="w", font=self.font(11), fill=COLORS["bad_text"])

        # Result with perfect play from the current position, from your view
        if self.outlook is not None:
            text, color = {1: ("Winning", COLORS["good"]), 0: ("Drawing", COLORS["secondary"]), -1: ("Losing", COLORS["bad_text"])}[self.outlook]
            c.create_text(16, top + CARDS["optimality"] - 22, text=text, anchor="w", font=self.font(13, bold=True), fill=color)


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
    root.title("Connect 4")
    Game(root, model, args.num_simulations)
    # the starting layout is the smallest one where nothing overlaps, only allow growing from it
    root.update_idletasks()
    root.minsize(root.winfo_reqwidth(), root.winfo_reqheight())
    root.mainloop()


if __name__ == "__main__":
    main()
