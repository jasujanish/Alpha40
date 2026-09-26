Goal
- Reimplement AlphaGo Zero for a simple game, Connect4

Network Architecture
Trunk
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

Policy Head
- Policy Convolution
    - Conv 1x1 2 filters, batchnorm, relu
    - (B,6,7,2)
- Flatten, Linear layer, Softmax
    - (B, 84)
    - 84 -> 7
    - (set full columns to have logits of -\inf so they don't get included in the softmax)
    - softmax to get move probs

Value Head
- Value Convolution
    - Conv 1x1, 1 filter, batchnorm, relu
    - (B, 6, 7, 1)
- Flatten, Linear layer
    - (B, 42)
    - 42 -> Relu -> 1 -> Tanh
- -1 = loss, 0 = draw, 1 = win