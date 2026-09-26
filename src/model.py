import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, filters=32, kernel_size = (3,3), stride=(1,1), padding=(1,1)):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size, stride, padding),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size, stride, padding),
            nn.BatchNorm2d(filters),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.layers(x)+x)


class AlphaC4Zero(nn.Module):
    def __init__(self, input_binary_chans=2, filters=32, kernel_size = (3,3), stride=(1,1), padding=(1,1), num_blocks = 8, hidden_dim = 32):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(input_binary_chans, filters, kernel_size, stride, padding),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            *[ResidualBlock(filters, kernel_size, stride, padding) for _ in range(num_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, 2, (1,1), stride),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(84,7),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(filters, 1, (1,1), stride),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(42, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, board):
        '''
        Input
        - board: (B,6,7,2), opponent/player binary planes of the board

        Output
        - moves: (B,7), probs for each move
        - value: (B,1), position value (close to 1 == likely win, close to -1 == likely lose)
        '''
        
        occupied = board.bool().any(dim=-1)     # (B, 6, 7)
        filled_cols = (occupied).all(dim=1)     # (B, 7)

        x = board.permute(0, 3, 1, 2).float()   # (B, 2, 6, 7)
        features = self.trunk(x)                # (B, 32, 6, 7)

        logits = self.policy_head(features)     # (B, 7)
        logits = logits.masked_fill(filled_cols, -torch.inf)
        probs = torch.softmax(logits, dim = -1)

        value = self.value_head(features)       # (B, 1)

        return probs, value
