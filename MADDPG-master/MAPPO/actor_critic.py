import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



# MAPPO 的 critic 网络是 以 联合观察作为输入， 输出单个Q值
# 而 actor 网络是 输入单个的观察维度，输出动作和对数概率, 这里直接继承SAC网络的actor
# define the actor network
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.action_shape = args.action_shape[agent_id]
        self.max_action = args.high_action
        self.min_action = args.low_action
        self.fc1 = nn.Linear(args.obs_shape[agent_id], 64)  # 18*64
        self.fc2 = nn.Linear(64, 64)  # Fully connected
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, self.action_shape)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 一层层连接，一共三层
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = np.int((self.max_action - self.min_action) / 2) * torch.tanh(self.action_out(x))
        return actions

    def get_actions(self, x, actions=None):
        # mean, log_std = self(x)
        # std = log_std.exp()
        # normal = torch.distributions.Normal(mean, std)  # Normal distribution
        # x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1)) 先对标准正态采样，再重新加上
        # # x_t, y_t : tensor(256, 2)
        # y_t = torch.tanh(x_t)
        # action = y_t * self.action_scale + self.action_bias  # 实际上就是把action规范到min——max区间中
        # log_prob = normal.log_prob(x_t)  # x值对应的对数概率，因为normal代表了一个正态分布的概率密度函数
        # # Enforcing Action Bound ??
        # log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        # log_prob = log_prob.sum(1, keepdim=True)
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias
        # entropy = normal.entropy().mean()   # may mistake
        logits = []
        out_actions = []
        action_log_probs = []
        entropys = []
        for i in range(self.action_shape):
            logits = self(x)
            probs = torch.distributions.Categorical(logits=logits)
            action = probs.sample()
            out_actions.append(action.float())
            action_log_prob = probs.log_prob(action)
            action_log_probs.append(action_log_prob)
            entropys.append(probs.entropy())
        if actions is None:
            actions = torch.cat(out_actions, -1)
            #log_prob = probs.log_prob(actions)
        action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
        log_prob = action_log_probs
        entropy = entropys
        return actions, log_prob, entropy

    # def evaluate_action(self, observation, action):
    #     """
    #     :param action: (torch.tensor) actions whose entropy and log probability to evaluate.
    #     dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
    #     action_log_probs: (torch.Tensor) log probabilities of the input actions.
    #
    #
    #     """
    #     action_log_probs = []
    #     dist_entropy = []
    #     actor_features = self.get_actions(observation)
    #
    #     # available_actions = ?
    #     # action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
    #     #                                                            action, available_actions=None,
    #     #                                                            active_masks=
    #     #                                                            active_masks if self._use_policy_active_masks
    #     #                                                            else None)
    #
    #     return action_log_probs, dist_entropy


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        self.centralized_input = args.centralized_input
        self.args = args
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

    def forward(self, state):
        if self.centralized_input:
            # state = torch.cat(state, dim=1)  # 将state中的元素拼接，按照行并排的方式，得到联合状态
            state = torch.cat(state, dim=1).squeeze(0)
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
