import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

seq_length = 10
# define the actor network
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.high_action
        self.min_action = args.low_action
        self.input_dim = args.obs_shape[agent_id]
        self.sequence_length = seq_length  # 新增的序列时间步跨度，比如10

        # LSTM层：输入是10维特征，输出是64维的隐藏状态
        self.rnn_hidden_size = 64  # 可调整的隐藏层大小
        self.lstm = nn.LSTM(self.input_dim, self.rnn_hidden_size, batch_first=True)

        # LSTM后的全连接层
        self.fc2 = nn.Linear(self.rnn_hidden_size, 64)
        self.fc3 = nn.Linear(64, 64)


        self.action_out = nn.Linear(64, args.action_shape[agent_id])

    def forward(self, x, hidden_state=None):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x, hidden_state = self.lstm(x, hidden_state)  # LSTM处理序列数据

        x = F.relu(self.fc2(x[:, -1, :]))  # 只取最后一个时间步的隐状态
        x = F.relu(self.fc3(x))
        actions = ((self.max_action-self.min_action)/2) * torch.tanh(self.action_out(x))  # 控制在[-1,1]*max_action
        return actions, hidden_state


    def get_input_dim(self):
        return self.input_dim

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        self.args = args
        self.input_dim = sum(args.obs_shape) + sum(args.action_shape)
        # LSTM层：用于处理状态和动作序列，输入维度为状态+动作的总和
        self.rnn_hidden_size = 64

        self.lstm = nn.LSTM(self.input_dim, self.rnn_hidden_size, batch_first=True)
        # 也就是说输入是两者的拼接
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)
        # 输出的是Q

    def forward(self, state, action, hidden_state=None):
        # 拼接状态和动作，形成联合输入 (batch_size, sequence_length, input_dim)
        state = torch.cat(state, dim=-1)  # 维度上拼接状态
        action = torch.cat(action, dim=-1)  # 维度上拼接动作
        x = torch.cat([state, action], dim=-1)  # 状态和动作拼接
        if x.ndim == 2:
            x = x.unsqueeze(1)
        # 输入到LSTM，得到每个时间步的隐藏状态
        x, hidden_state = self.lstm(x, hidden_state)

        # 使用最后一个时间步的LSTM输出作为全连接层的输入
        x = F.relu(self.fc2(x[:, -1, :]))  # 只取最后一个时间步的隐藏状态
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value, hidden_state  # 输出Q值和隐状态

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
        actions, _ = self.actor(obs_ac)
        q, _ = self.critic([obs], [act])
        return actions, q