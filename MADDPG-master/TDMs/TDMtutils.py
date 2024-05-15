import torch
from replay_buffer import Buffer
import numpy as np


# Relabel 重要一步是从状态空间中随机抽样出目标位置以及随机抽样τ
class Relable:
    def __init__(
            self,
            args,
            observation,
            target_state,
            action,
            next_observation,
            tao_max,
    ):
        self.args = args

    def sample_state(self, generate_sample_shape, state_dim, state_bound, continuous_space=True):
        """
        state_dim: describe the dimension of state vector.  (scalar number n)
        state_bound: should correspond to the bound of each dimension of state. (list or ndarray,shape must be n*2)
        continuous_space: state sample is continuous. (boolean)
        generate_sample_shape: the size or the number of samples u want to generate (list  [n,m])

        this method will generate state samples with uniform distribution according to the bound.
        """
        assert state_dim == len(state_bound), "state size is not equal to size of constraints."
        state_store = []
        if continuous_space:
            for row in range(generate_sample_shape[0]):
                for dim in range(state_dim):
                    temp_state = np.random.uniform(state_bound[dim][0], state_bound[dim][1], (generate_sample_shape[1]))
                    state_store.append(temp_state)
        state_store = np.array(state_store).reshape(tuple(generate_sample_shape))
        assert np.array(state_store).shape == generate_sample_shape,"error state sample processing"
        return state_store

    def sample_tao(self, number):


class TDM_ReplayBuffer(Buffer):
    def __init__(self, args):
        super().__init__(args)

    def store_episode(self, o, u, r, o_next):
