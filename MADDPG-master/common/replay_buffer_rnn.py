import threading
import numpy as np


class Buffer_RNN:
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
    # def sample(self, batch_size):
    #     temp_buffer = {}
    #     idx = np.random.randint(0, self.current_size, batch_size)
    #     for key in self.buffer.keys():
    #         temp_buffer[key] = self.buffer[key][idx]
    #     return temp_buffer

    # sample the data from the replay buffer
    def sample(self, batch_size, length=10):
        temp_buffer = {key: [] for key in self.buffer.keys()}
        # 随机选择一个起始索引，确保能提取到连续的10个时间步
        max_start_idx = self.current_size - length  # 计算最大起始索引
        if max_start_idx < 0:  # 如果当前buffer没有足够的样本，返回空
            return None
        for _ in range(batch_size):
            # 随机选择一个起始索引
            start_idx = np.random.randint(0, max_start_idx)

            # 对 buffer 中每个 agent 或每种类型的样本进行采样
            for key in self.buffer.keys():
                # 提取从 start_idx 开始的10个连续时间步的数据
                extracted_data = self.buffer[key][start_idx:start_idx + length]

                # 将提取的数据追加到 temp_buffer 中
                temp_buffer[key].append(extracted_data)
        # 类型为batch_size, 10, xxx, 这种带有序列形式的
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx