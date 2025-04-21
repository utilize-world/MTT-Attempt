import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 接受本地o和本地a，输出偏移控制量

class ActorSafeNet(nn.Module):
    def __init__(self, args):
        super(ActorSafeNet, self).__init__()
        self.max_action = args.high_action
        self.args = args
        self.input_dim = args.obs_shape[0] + args.action_shape[0]
        self.fc1 = nn.Linear(self.input_dim, 64)  # 输入维度是观测维度和动作维度的和
        # 也就是说输入是两者的拼接
        self.fc3 = nn.Linear(64, 64)
        self.delta_action_out = nn.Linear(64, 2)

    def forward(self, obs, action_policy):
        x = torch.cat([obs, action_policy], dim=1)  # 拼接状态和动作得到联合状态-动作对，也就是输入到critic中的是联合状态和联合动作
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        delta_action = F.tanh(self.delta_action_out(x))
        return action_policy + delta_action
