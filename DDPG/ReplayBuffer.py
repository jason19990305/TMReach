from collections import deque 
import numpy as np 
import torch




class ReplayBuffer:
    def __init__(self, args):
        self.max_length = args.buffer_size
        self.s = deque(maxlen = self.max_length)
        self.a = deque(maxlen = self.max_length)
        self.r = deque(maxlen = self.max_length)
        self.s_ = deque(maxlen = self.max_length)
        self.dw = deque(maxlen = self.max_length)
        self.done = deque(maxlen = self.max_length)
        self.count = 0

    def store(self, s, a, r, s_, done):
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.s_.append(s_)
        self.done.append([done])
        if self.count <= self.max_length:
            self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(np.array(self.s), dtype=torch.float)
        a = torch.tensor(np.array(self.a), dtype=torch.float)
        r = torch.tensor(np.array(self.r), dtype=torch.float)
        s_ = torch.tensor(np.array(self.s_), dtype=torch.float)
        done = torch.tensor(np.array(self.done), dtype=torch.float)

        return s, a, r, s_, done
