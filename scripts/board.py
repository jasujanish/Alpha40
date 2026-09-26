import torch

DEFAULT_HEIGHT = 6
DEFAULT_WIDTH = 7
WIN_LENGTH = 4

class Board():
    def __init__(self, height = DEFAULT_HEIGHT, width = DEFAULT_WIDTH, win_length = WIN_LENGTH, device = "cpu"):
        self.height = height
        self.width = width
        self.win_length = WIN_LENGTH
        self.pieces = torch.zeros((height, width), device=device, requires_grad=False)

    def add_stone(self, col, player):
        assert (0 <= col and col < self.width), "col out of order"
        available = (self.pieces[:, col]==0).nonzero(as_tuple = True)[0]
        assert (available.numel() > 0), "full col"
        self.pieces[available[-1], col] = player

    def get_moves(self):
        return self.pieces[0] == 0

    def moves_left(self):
        return (self.pieces[0] == 1).any()
    
    def has_won(self, player):
        mask = self.pieces == player
        k = self.win_length
        won = torch.zeros((), dtype=torch.bool, device=mask.device)

        # Shape: (height, width - k + 1, k)
        if self.width >= k:
            horizontal = mask.unfold(1, k, 1)
            won |= horizontal.all(dim=-1).any()

        # Shape: (height - k + 1, width, k)
        if self.height >= k:
            vertical = mask.unfold(0, k, 1)
            won |= vertical.all(dim=-1).any()

        # Shape: (height - k + 1, width - k + 1, k, k)
        if self.height >= k and self.width >= k:
            windows = mask.unfold(0, k, 1).unfold(1, k, 1)

            down_right = windows.diagonal(dim1=-2, dim2=-1)
            down_left = windows.flip(-1).diagonal(dim1=-2, dim2=-1)

            won |= down_right.all(dim=-1).any()
            won |= down_left.all(dim=-1).any()

        return won

    def __str__(self):
        return str(self.pieces)
            