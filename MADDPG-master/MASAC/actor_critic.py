import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 对于SAC, Q网络输入S,A，输出Q，V网络输入为S，输出V，Actor网络的输入为O，输出比较复杂：action, mean, log_std, log_prob, entropy, std,
#             mean_action_log_prob, pre_tanh_value,
#             关键的是前四个值，包括action，均值，对数std和对数概率。
#  总共四个网络，A, Q, V, Target_V.
# define the actor network

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    """
       actor网络的输入为每个agent的观察维度，输出为均值与对数标准差，
       """
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.high_action
        self.min_action = args.low_action
        self.actor_id = agent_id
        self.fc1 = nn.Linear(args.obs_shape[agent_id], 64)  # 18*64
        self.fc2 = nn.Linear(64, 64)            # Fully connected
        self.fc3 = nn.Linear(64, 64)
        self.fc_logstd = nn.Linear(64, args.action_shape[agent_id])    #
        self.fc_mean = nn.Linear(64, args.action_shape[agent_id])   # 均值输出层

        self.register_buffer(
            "action_scale",
            torch.tensor((self.max_action - self.min_action) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((self.max_action + self.min_action) / 2.0, dtype=torch.float32)
        )
        # 这里实际上是对action的广度和偏差进行保存，这是考虑连续动作的情况
        # 这里修改的点不知道有没有问题，以下是原文
        # def __init__(self, env: ParallelEnv):
        #     super().__init__()
        #     single_action_space = env.action_space(env.agents[0])
        #     single_observation_space = env.observation_space(env.agents[0])
        #     # Local state, agent id -> ... -> local action
        #     self.fc1 = nn.Linear(np.array(single_observation_space.shape).prod() + 1, 256)
        #     self.fc2 = nn.Linear(256, 256)
        #     self.fc_mean = nn.Linear(256, np.prod(single_action_space.shape))
        #     self.fc_logstd = nn.Linear(256, np.prod(single_action_space.shape))
        #     # action rescaling
        #     self.register_buffer(
        #         "action_scale",
        #         torch.tensor((single_action_space.high - single_action_space.low) / 2.0, dtype=torch.float32)
        #     )
        #     self.register_buffer(
        #         "action_bias",
        #         torch.tensor((single_action_space.high + single_action_space.low) / 2.0, dtype=torch.float32)
        #     )

    def forward(self, x):
        x = F.relu(self.fc1(x))     # 一层层连接，一共三层
        x = F.relu(self.fc2(x))

        log_std = self.fc_logstd(x)
        # log_std: tensor(256,2)
        log_std = torch.tanh(log_std)
        mean = self.fc_mean(x)
        # mean: tensor(256,2)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_actions(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)  #   Normal distribution
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1)) 先对标准正态采样，再重新加上
        # x_t, y_t : tensor(256, 2)
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias  # 实际上就是把action规范到min——max区间中
        log_prob = normal.log_prob(x_t) # x值对应的对数概率，因为normal代表了一个正态分布的概率密度函数
        # Enforcing Action Bound ??
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class Q_net(nn.Module):
    def __init__(self, args):
        super(Q_net, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 64)  # 输入维度是观测维度和动作维度的和,也就是联合动作和联合状态
        # 也就是说输入是两者的拼接
        # Q 网络的输出可以是所有agent的动作空间维度之和，以及加上全局状态，作为一个centralized critic
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


class V_net(nn.Module):
    """
    Vnet是用来拟合softQ网络中的目标项，也就是贝尔曼方程的第二项，仍受到γ影响？

    """
    def __init__(self, args, ):
        super(V_net, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(sum(args.obs_shape), 64)  # 输入维度是观测维度
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)
        # 输出的是V

    def forward(self, state, action):
        state = torch.cat(state, dim=1)  # 将state中的元素拼接，按照行并排的方式，得到联合状态
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        v_value = self.q_out(x)
        return v_value
