from typing import Dict, List, Tuple
import collections
import random


class Buffer(object):
    def __init__(self, max_length: int) -> None:
        super().__init__()
        self.max_len = max_length
        self.buffer = collections.deque(maxlen=max_length)
    
    def push(self, transition: Tuple) -> None:
        self.buffer.append(transition)

    def push_batch(self, trans_batch: List[Tuple]) -> None:
        for item in trans_batch:
            self.push(item)

    def sample(self, batch_size: int) -> List[Tuple, Tuple, Tuple, Tuple]:
        data = random.sample(self.buffer, batch_size)
        return list(zip(*data))