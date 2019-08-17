
import numpy as np




class Memory:

    def __init__(self, state_dim, memory_size):
        self.state_dim = state_dim
        self.memory_size = memory_size
        self._empty_state = np.zeros((1, self.state_dim))

        self.DB = np.rec.recarray(self.memory_size, dtype=[
            ("prev_state", np.float32, self.state_dim),
            ("next_state", np.float32, self.state_dim),
            ("action", np.int32),
            ("reward", np.float32),
        ])
        self.clear()

    def clear(self):
        """Remove all entries from the DB."""

        self.index = 0
        self.n_items = 0
        self.full = False

    def remember(self, prev_state, next_state, action, reward):
        """Store new samples in the DB."""

        n = prev_state.shape[0]
        if self.index + n > self.memory_size:
            self.full = True
            l = self.memory_size - self.index
            if l > 0:
                self.remember(prev_state[:l], next_state[:l], action[:l], reward[:l])
            self.index = 0
            if l < n:
                self.remember(prev_state[l:], next_state[l:], action[l:], reward[l:])
        else:
            v = self.DB[self.index: self.index + n]
            v.prev_state = prev_state
            v.next_state = next_state
            v.action = action
            v.reward = reward
            self.index += n

        self.n_items = min(self.n_items + n, self.memory_size)

    def sample(self, sample_size=None):
        """Get a random sample from the DB."""
        if self.full:
            db = self.DB
        else:
            db = self.DB[:self.index]

        if (sample_size is None) or (sample_size > self.n_items):
            return db
        else:
            return np.rec.array(np.random.choice(db, sample_size, False))