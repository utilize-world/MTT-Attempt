import numpy as np
import torch
import os
from maddpg.maddpg import MADDPG


class Agent:
    def __init__(self, agent_id, args):
        self.args = args    # 这个的成员的幅值在make_env中传递过来了，所以更改make_env的参数即可
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(self.args.low_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)  # 给第0位添加一维向量，本来o是(18,)，现在是tensor(1,observation shape)
            pi = self.policy.actor_network(inputs).squeeze(0)   # 压缩第0位的一维向量,这里policy的输出是tensor(64,action_shape),最后变成(1,action_shape)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            noise = 1.2 * noise_rate * ((self.args.high_action-self.args.low_action)/2) * np.random.randn(*u.shape) # gaussian noise
            u += noise
            u = np.clip(u, np.int(-(self.args.high_action-self.args.low_action)/2),
                        np.int((self.args.high_action-self.args.low_action)/2))
            # np.clip(a, a_min, a_max, out=None) 限制其值在amin，amax之间
            u += np.int((self.args.high_action-self.args.low_action)/2)     # 要限制其动作在0到high_action内
        return u.copy()

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

