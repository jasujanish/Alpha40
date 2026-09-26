import collections
import random

class ReplayBuffer():
    def __init__(self, buffer_size, batch_size):
        '''Initialize replay memory as a deque'''
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = collections.deque(maxlen=buffer_size)

    def add(self, entry):
        '''Add to buffer, removes oldest entry if we are at buffer_size'''
        self.buffer.append(entry)

    def sample(self):
        '''Sample a minibatch'''
        if len(self.buffer) < self.batch_size:
            return None
        return random.sample(self.buffer, self.batch_size)
    
    def __len__(self):
        return len(self.buffer)
