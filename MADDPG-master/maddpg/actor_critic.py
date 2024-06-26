import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# define the actor network
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.high_action
        self.min_action = args.low_action
        self.fc1 = nn.Linear(args.obs_shape[agent_id], 64)  # 18*64
        self.fc2 = nn.Linear(64, 64)            # Fully connected
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, args.action_shape[agent_id])

    def forward(self, x):
        x = F.relu(self.fc1(x))     # 一层层连接，一共三层
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = np.int((self.max_action-self.min_action)/2) * torch.tanh(self.action_out(x))  # 控制在[-1,1]*max_action

        return actions


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 64)  # 输入维度是观测维度和动作维度的和
        # 也就是说输入是两者的拼接
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)
        # 输出的是Q

    def forward(self, state, action):
        state = torch.cat(state, dim=1) # 将state中的元素拼接，按照行并排的方式，得到联合状态
        # 具体有关cat  https://blog.csdn.net/scar2016/article/details/121382717
        for i in range(len(action)):
            action[i] /= self.max_action        # 这一步是把action[i] ”归一化“，限制到[-1,1]
        action = torch.cat(action, dim=1)       # 拼接得到联合动作
        x = torch.cat([state, action], dim=1)   # 拼接状态和动作得到联合状态-动作对，也就是输入到critic中的是联合状态和联合动作
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
