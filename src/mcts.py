import torch

class Node:
    '''Node in MCTS tree'''
    def __init__(self, board, to_play=1): 
        if to_play not in (1, -1):
            raise ValueError("to_play must be 1 or -1")
        self.state = board.copy()
        self.to_play = to_play
        self.moves = self.state.get_moves().nonzero(as_tuple=True)[0].tolist()
        self.expanded = False
        self.priors = {} # P(s,a), populated by expand
        
        self.children = {}

        self.visit_counts = {} # N(s,a)
        self.accum_results = {} # W(s,a)
        # Q(s,a) = W(s,a)/N(s,a)

    def is_terminal(self):
        return self.state.full() or self.state.has_won(1) or self.state.has_won(-1)

    def result(self):
        if self.state.has_won(1):
            return 1
        
        if self.state.has_won(-1):
            return -1

        return 0
    
class MCTS:   
    '''MCTS tree and associated search methods, we use 1 tree per game'''
    def __init__(self, board, model, to_play=1, c_puct = 1.0, device = 'cpu'):
        self.root_node = Node(board, to_play)
        self.c_puct = c_puct
        self.device = device
        self.model = model

    def search(self, num_simulations):
        '''Run num_simulations searches from the current node, return most visited move'''
        assert num_simulations >= 2
        for i in range(num_simulations):
            self.select(self.root_node)

        best_move = None
        best_visits = 0
        for move in self.root_node.moves:
            if self.root_node.visit_counts.get(move, 0) > best_visits:
                best_move = move
                best_visits = self.root_node.visit_counts.get(move, 0)
        return best_move
        # most visited is a simplification of alpha go zero, it uses visit count sampling with root exploration noise!!!!

    def select(self, node):
        '''Selects a move, runs one simulation till terminal leaf'''
        if node.state.has_won(node.to_play):
            return 1.0
        if node.state.has_won(-node.to_play):
            return -1.0
        if not node.moves:
            return 0.0

        if not node.expanded:
            return float(self.expand(node))

        total_visits = sum(node.visit_counts.values())

        def puct(move):
            visits = node.visit_counts[move]
            mean_value = node.accum_results[move] / visits if visits else 0.0
            bonus = self.c_puct * node.priors[move] * total_visits**0.5 / (1 + visits)
            return mean_value + bonus

        # Use the prior to break ties, including when every visit count is zero.
        move = max(node.moves, key=lambda a: (puct(a), node.priors[a]))

        if move not in node.children:
            child = Node(node.state, to_play=-node.to_play)
            child.state.add_stone(move, node.to_play)
            child.moves = child.state.get_moves().nonzero(as_tuple=True)[0].tolist()
            node.children[move] = child

        # The child returns its player's value, so reverse the perspective.
        value = -self.select(node.children[move])

        # Back up once per traversed edge.
        node.visit_counts[move] += 1
        node.accum_results[move] += value
        return value

    def expand(self, node):
        '''Evaluate one leaf and initialize its legal edges without visiting them.'''
        if node.state.has_won(node.to_play):
            return 1.0
        if node.state.has_won(-node.to_play):
            return -1.0
        if not node.moves:
            return 0.0

        # (batch, height, width, opponent/player planes) = (1,6,7,2)
        pieces = node.state.pieces
        board = torch.stack((pieces == -node.to_play, pieces == node.to_play), dim=-1).unsqueeze(0) # (1,6,7,2)
        board = board.to(self.device)

        # Search is no grad
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                probs, value = self.model(board)
        finally:
            self.model.train(was_training)

        for move in node.moves:
            node.priors[move] = probs[0, move].item()
            node.visit_counts[move] = 0
            node.accum_results[move] = 0.0

        node.expanded = True
        return value.item()

    def advance(self, move):
        '''move root node based on opponent move'''
        if move not in self.root_node.moves or self.root_node.is_terminal():
            return False
        
        if move in self.root_node.children:
            self.root_node = self.root_node.children[move]
        else:
            board = self.root_node.state.copy()
            board.add_stone(move, self.root_node.to_play)
            self.root_node = Node(board, to_play=-self.root_node.to_play)
        
        return True
