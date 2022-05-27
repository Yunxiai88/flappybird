import random
import numpy as np
from collections import deque

class ReplayMemory:
    def __init__(self, capacity = 5000):
        self.memory   = []
        self.capacity = capacity
        self.D        = deque()
    
    def add(self, experience):
        experience = np.reshape(np.array(experience), [1, 5])
        if len(self.memory) + len(experience) >= self.capacity:
            del self.memory[0:len(experience)]
        
        self.memory.extend(experience)
    
    def sample(self, batch_size):
        data = random.sample(self.memory, batch_size)

        status      = [d[0] for d in data]
        action      = [d[1] for d in data]
        reward      = [d[2] for d in data]
        next_status = [d[3] for d in data]
        terminal    = [d[4] for d in data]

        return status, action, reward, next_status, terminal
    
    # no use
    def add1(self, experience):
        self.D.append(experience)
        if len(self.D) > self.capacity:
            self.D.popleft()
    
    def sample1(self, batch_size):
        minibatch = random.sample(self.D, batch_size)

        state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
        state_t = np.concatenate(state_t)
        state_t1 = np.concatenate(state_t1)

        return state_t, action_t, reward_t, state_t1, terminal