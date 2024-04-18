import threading
import numpy as np


class Buffer:
    def __init__(self, args):
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
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, o, u, r, o_next):
        idxs = self._get_storage_idx(inc=1)  # 以transition的形式存，每次只存一条经验
        for i in range(self.args.n_agents):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = o[i]
                self.buffer['u_%d' % i][idxs] = u[i]
                self.buffer['r_%d' % i][idxs] = r[i]
                self.buffer['o_next_%d' % i][idxs] = o_next[i]
    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx

# class MAPPO_buffer(Buffer):
#     def __init__(self, args):
#         super().__init__(args)
#         self.size = args.buffer_size
#         self.args = args
#
#         self.buffer = dict()
#         self.current_size = 0
#         for i in range(self.args.n_agents):
#             self.buffer['o_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
#             self.buffer['u_%d' % i] = np.empty([self.size, self.args.action_shape[i]])
#             self.buffer['r_%d' % i] = np.empty([self.size])
#             self.buffer['o_next_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
#             self.buffer['adv_func_%d' % i] = np.empty([self.size, 1])   # 优势函数应为一个值
#             self.buffer['old_u_%d' % i] = np.empty([self.size, self.args.action_shape[i]])
#         self.lock = threading.Lock()
#
#
#     def sample(self, batch_size):
#         return Buffer.sample(self, batch_size)
#
#     def _get_storage_idx(self, inc=None):
#         return Buffer._get_storage_idx(self, inc)
#
#
#     def PPO_store_episode(self, o, u, r, o_next, adv_func, old_u):
#         idxs = self._get_storage_idx(inc=1)  # 以transition的形式存，每次只存一条经验
#         for i in range(self.args.n_agents):
#             with self.lock:
#                 self.buffer['o_%d' % i][idxs] = o[i]
#                 self.buffer['u_%d' % i][idxs] = u[i]
#                 self.buffer['r_%d' % i][idxs] = r[i]
#                 self.buffer['o_next_%d' % i][idxs] = o_next[i]
#                 self.buffer['adv_func_%d' % i][idxs] = adv_func[i]
#                 self.buffer['old_u_%d' % i][idxs] = old_u[i]