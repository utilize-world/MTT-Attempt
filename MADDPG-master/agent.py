import numpy as np
import torch
import os
from maddpg.maddpg import MADDPG
from MASAC.MASAC import MASAC


class Agent:
    def __init__(self, agent_id, args, policy, algorithm):
        self.args = args    # 这个的成员的幅值在make_env中传递过来了，所以更改make_env的参数即可
        self.agent_id = agent_id
        self.policy = policy
        self.algorithm = algorithm

    def select_action(self, o, noise_rate, epsilon, global_info=None):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(self.args.low_action, self.args.high_action, self.args.action_shape[self.agent_id])
        elif self.algorithm == "MADDPG":
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)  # 给第0位添加一维向量，本来o是(18,)，现在是tensor(1,observation shape)
            pi = self.policy.actor_network(inputs).squeeze(0)   # 压缩第0位的一维向量,这里policy的输出是tensor(64,action_shape),最后变成(1,action_shape)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            noise = 2 * noise_rate * ((self.args.high_action-self.args.low_action)/2) * np.random.randn(*u.shape) # gaussian noise
            u += noise
            u = np.clip(u, np.int(-(self.args.high_action-self.args.low_action)/2),
                        np.int((self.args.high_action-self.args.low_action)/2))
            # np.clip(a, a_min, a_max, out=None) 限制其值在amin，amax之间
            u += np.int((self.args.high_action-self.args.low_action)/2)     # 要限制其动作在0到high_action内

        elif self.algorithm == "MASAC" or self.algorithm == "MAPPO":
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi2, prob, _ = self.policy.policy.get_actions(inputs)
            pi = pi2
            pi = pi.squeeze(0)
            u = pi.cpu().numpy()
            u = np.clip(u, np.int(-(self.args.high_action - self.args.low_action) / 2),
                        np.int((self.args.high_action - self.args.low_action) / 2))
            # np.clip(a, a_min, a_max, out=None) 限制其值在amin，amax之间
            u += np.int((self.args.high_action - self.args.low_action) / 2)
            if self.algorithm == "MAPPO":
                # inputs = torch.cat([inputs, global_info], dim=1)
                # global_info = torch.tensor(global_info).unsqueeze(0)
                global_info = [torch.tensor(row, dtype=torch.float32).unsqueeze(0) for row in global_info]
                values = self.policy.critic(global_info)
                return u.copy(), pi2, prob, values
        else:
            print("error arguments input")
            return None
        return u.copy()

    def learn(self, transitions, other_agents, algorithm,
              obs=None,
              next_obs=None,
              values=None, dones=None,
              actions=None, logprobs=None,

              rewards=None, nextdone=None, time_steps=None):
        if algorithm == "MADDPG":
            self.policy.train(transitions, other_agents)
        elif algorithm == "MASAC":
            self.policy.train(transitions, self.agent_id)
        elif algorithm == "MAPPO":
            self.policy.train(obs, next_obs, values, dones, actions, logprobs, rewards, nextdone, self.agent_id, time_steps)
        else:
            print("error arguments input")
