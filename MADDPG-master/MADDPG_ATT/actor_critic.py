import torch
import torch.nn as nn
import torch.nn.functional as F
from GlobalAtt import GlobalAttention
from Local import LocalAttention

# LOG_STD_MAX = 2
# LOG_STD_MIN = -20

class Actor(nn.Module):
    def __init__(self, args, agent_id=0):
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

class Q_net(nn.Module):
    def     __init__(self, args, init_w=3e-3):
        super(Q_net, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(256, 128)  # 输入维度是观测维度和动作维度的和,也就是联合动作和联合状态
        self.q_out = nn.Linear(128, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.layer_norm = nn.LayerNorm(128)
        # -------------- attention module
        self.globalATT = GlobalAttention(2)
        self.localAtt = LocalAttention(2)
        # --------------
        self.q_out.weight.data.uniform_(-init_w, init_w)
        self.q_out.bias.data.uniform_(-init_w, init_w)
        # 输出的是Q

    def forward(self, state_own, state_others, action_own, action_others):
        # 这里传入的直接就是ei，ej，所谓的ei和ej分别是通过了global attention之后的输出
        ei, ej = self.get_attention_representation(state_own, state_others, action_own, action_others)
        # 得到两个batch_size, 8, 128

        # # 注意，这里的state和action应为(batch_size, state_repre, 2),(batch_size, 1, 2)
        # local_state = torch.cat((state_own, action_own), dim=-2) # 将state中的元素拼接，得到自身的信息表示
        # other_state = torch.cat((state_others, action_others), dim=-2) # 将其它agent所有元素拼接
        # # 具体有关cat  https://blog.csdn.net/scar2016/article/details/121382717
        # # TODO: 这里的localATT的输入有两项，一项是self的信息，一项是decomposed from self，具体不知道是怎么表达的，暂且让两者一样
        # local_att_weight, local_repre = self.localAtt(state_own, state_own)
        # # 这里的local_repre为ei，形状为(batch, state_repre, 2)  与输入一致
        x = torch.cat((ei, ej), dim=-1) # 最后一维进行拼接，就变成了(batch, sequence_len, 256)
        x = F.relu(self.fc1(x))
        # q_value = self.layer_norm(x)
        q_value = self.q_out(x)     # (batch_size, sequence_len, 1)

        # Global average pooling to get (batch_size, 1)
        # TODO: 不知道这一步是否正确,把第二维的值取平均，就消除掉了第二维
        final_output = q_value.mean(dim=1)  # (batch_size, 1)
        return final_output

    def get_attention_representation(self, state_own, state_others, action_own, action_others):
        # # TODO: 这里的localATT的输入有两项，一项是self的信息，一项是decomposed from self，具体不知道是怎么表达的，暂且让两者一样
        # action_own是一个tensor，(batch, 1, 2)
        # action_other是一个list, 其中包含n-1个(batch, 1, 2)tensor
        # state dim 是指观察空间中，有多少个二维数据
        local_att_weight, ei = self.localAtt(state_own, state_own)
        # 将state拆分成n个相同格式的列表,！！！这一步是在state_others为拼接后的列表为前提，一般state_others应该是多个tensor组成的list
        # split_list = [state_others[i:i+len(state_others)] for i in range(0, len(state_others), len(state_others)*state_dim)]
        # split_list = [[chunk[i:i+state_dim] for i in range(0, len(chunk), state_dim)] for chunk in split_list]
        # 即n个(batch_size, 7, 2)
        eloc = []
        for other in state_others:
            _, ej = self.localAtt(other, other)
            eloc.append(ej)
        # 这里得到的ej也是batch_size, 7, 2
        # 然后把ej和a进行cat，变成batch_size,8,2,再把n个都cat在一起变成batch，8n, 2，就得到了eother
        eloc_i = torch.cat((ei, action_own), dim=-2)
        for i, ej in enumerate(eloc):
            eloc[i] = torch.cat((ej, action_others[i]), dim=-2) # 拼成batch_size, 8, 2
        eother = torch.cat(eloc, dim=-2)    # 拼成batch, 8(n-1), 2

        ei, ej = self.globalATT(eloc_i, eother) # 经过globalATT得到了两者表达
        return ei, ej

class Wrapper(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        # build policy and value functions
        self.actor = Actor(args)
        self.critic = Q_net(args)

    def forward(self, x, state_own, state_others, action_own, action_others):

        # Perform a forward pass through all the networks and return the result
        actions = self.actor(x)
        q = self.critic(state_own, state_others, action_own, action_others)
        return actions, q