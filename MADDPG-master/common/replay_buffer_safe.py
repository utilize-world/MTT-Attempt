import threading
import numpy as np
from replay_buffer import Buffer


class replay_buffer_safe(Buffer):
    def __init__(self, args):
        super().__init__(args)
        self.size = args.buffer_size
        self.args = args
        # memory management
        self.current_size = 0
        # create the buffer to store info
        self.buffer = dict()
        for i in range(self.args.n_agents):
            self.buffer['o_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
            self.buffer['u_%d' % i] = np.empty([self.size, self.args.action_shape[i]])
            self.buffer['r_%d' % i] = np.empty([self.size])
            self.buffer['o_next_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
            self.buffer['u_safe_%d' % i] = np.empty([self.size, self.args.action_shape[i]])
        # thread lock
        self.lock = threading.Lock()

    def store_episode_safe(self, o, u, r, o_next, u_safe):
        idxs = self._get_storage_idx(inc=1)  # 以transition的形式存，每次只存一条经验
        for i in range(self.args.n_agents):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = o[i]
                self.buffer['u_%d' % i][idxs] = u[i]
                self.buffer['r_%d' % i][idxs] = r[i]
                self.buffer['o_next_%d' % i][idxs] = o_next[i]
                self.buffer['u_safe_%d' % i][idxs] = u_safe[i]



