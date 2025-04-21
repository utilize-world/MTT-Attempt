import torch
import torch.nn as nn
import torch.nn.functional as F

class Qnet(nn.Module):
    def __init__(self, args, action_dim):
        super(Qnet, self).__init__()
        self.max_action = args.high_action
        self.args = args
        self.input_dim = args.obs_shape[0]
        self.fc1 = nn.Linear(self.input_dim, 64)  # 输入维度是观测维度和动作维度的和
        # 也就是说输入是两者的拼接
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, action_dim)
        # 输出的是Q

    def forward(self, state):
        # state = torch.cat(state, dim=1) # 将state中的元素拼接，按照行并排的方式，得到联合状态
        # action = torch.cat(action, dim=1)       # 拼接得到联合动作  同样action也是2*tensor(256*2)拼接后就是联合动作
        # x = torch.cat([state, action], dim=1)   # 拼接状态和动作得到联合状态-动作对，也就是输入到critic中的是联合状态和联合动作
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value