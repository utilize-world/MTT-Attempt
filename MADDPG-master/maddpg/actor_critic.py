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
        self.input_dim = args.obs_shape[agent_id]
        self.fc1 = nn.Linear(self.input_dim, 64)  # 18*64
        self.fc2 = nn.Linear(64, 64)            # Fully connected
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, args.action_shape[agent_id])

    def forward(self, x):
        x = F.relu(self.fc1(x))     # 一层层连接，一共三层
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = ((self.max_action-self.min_action)/2) * torch.tanh(self.action_out(x))  # 控制在[-1,1]*max_action


        return actions

    def get_input_dim(self):
        return self.input_dim

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        self.args = args
        self.input_dim = sum(args.obs_shape) + sum(args.action_shape)
        self.fc1 = nn.Linear(self.input_dim, 64)  # 输入维度是观测维度和动作维度的和
        # 也就是说输入是两者的拼接
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)
        # 输出的是Q

    def forward(self, state, action):
        state = torch.cat(state, dim=1) # 将state中的元素拼接，按照行并排的方式，得到联合状态
        # 2*256*19 -> 256*38 为什么？因为torch.cat传入的参数实际上是[tensor(256*19),tensor(256*19)],拼接的是中间的所有tensor，即dim=1，就是对于19那一维度拼接，即变成了状态的拼接
        # 具体有关cat  https://blog.csdn.net/scar2016/article/details/121382717
        for i in range(len(action)):
            action[i] /= self.max_action        # 这一步是把action[i] ”归一化“，限制到[-1,1]
        action = torch.cat(action, dim=1)       # 拼接得到联合动作  同样action也是2*tensor(256*2)拼接后就是联合动作
        x = torch.cat([state, action], dim=1)   # 拼接状态和动作得到联合状态-动作对，也就是输入到critic中的是联合状态和联合动作
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value

    def get_input_dim(self):
        return sum(self.args.obs_shape), sum(self.args.action_shape)

# 只是用于tensorboard画出网络结构的类
class Wrapper(nn.Module):
    def __init__(
        self,
        args,
        agent_id,
    ):
        super().__init__()
        # build policy and value functions
        self.actor = Actor(args, agent_id)
        self.critic = Critic(args)

    def forward(self, obs, act, obs_ac):

        # Perform a forward pass through all the networks and return the result
        actions = self.actor(obs_ac)
        q = self.critic([obs], [act])
        return actions, q