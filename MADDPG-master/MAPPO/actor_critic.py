import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..MASAC.MASAC import Actor as AC


# MAPPO 的 critic 网络是 以 联合观察作为输入， 输出单个Q值
# 而 actor 网络是 输入单个的观察维度，输出动作和对数概率, 这里直接继承SAC网络的actor
# define the actor network
class Actor(AC):
    def __init__(self, args, agent_id):
        super().__init__(args, agent_id)

    def forward(self, x):
        AC.forward(self, x)

    def get_actions(self, x):
        AC.get_actions(self, x)



class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        self.centralized_input = args.centralized_input
        # 如果centralized为true就是MAPPO，输入为joint obs, 否则就是IPPO,
        if self.centralized_input:
            self.fc1 = nn.Linear(sum(args.obs_shape), 64)  # 输入维度是观测维度和动作维度的和
        else:
            self.fc1 = nn.Linear(args.obs_shape[0], 64)
        # 也就是说输入是两者的拼接
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)
        # 输出的是Q

    def forward(self, state, action):
        if self.centralized_input:
            state = torch.cat(state, dim=1)  # 将state中的元素拼接，按照行并排的方式，得到联合状态
        # 其实进来的结构是agent_number*batch*observation,拼接够会变成batch*(2*observation)
        # 具体有关cat  https://blog.csdn.net/scar2016/article/details/121382717
        # for i in range(len(action)):
        #     action[i] /= self.max_action        # 这一步是把action[i] ”归一化“，限制到[-1,1]
        # action = torch.cat(action, dim=1)       # 拼接得到联合动作
        # x = torch.cat(state, dim=1)   # 拼接状态和动作得到联合状态-动作对，也就是输入到critic中的是联合状态和联合动作
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
