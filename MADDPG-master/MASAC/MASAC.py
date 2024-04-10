import torch
import os
from .actor_critic import Actor, Q_net


class MASAC:
    def __init__(self, args):
        self.args = args
        self.train_step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.policy = [policy_net(args, i) for i in range(args.n_agents)]  # 创建agent个数个actor网络
        self.q_lr = 1e-3

        self.qf1 = Q_net(args).to(self.device)
        self.qf2 = Q_net(args).to(self.device)

        self.qf1_t = Q_net(args)
        self.qf2_t = Q_net(args)

        self.qf1_t.load_state_dict(self.qf1.state_dict())
        self.qf2_t.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = torch.optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.q_lr)

    def train(self, transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)

        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])

        u_next = []
        real_o_next = o_next

        with torch.no_grad():
            for

class policy_net:
    """
    给出所有的policy，也就是所有的actor_network
    """

    def __init__(self, args, agent_id):
        self.id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        # 创建actor网络以及目标网络
        self.p_lr = 3e-4
        self.actor = Actor(args, agent_id).to(self.device)
        # self.actor_t = Actor(args, agent_id)
        self.alpha = args.alpha
        # self.actor_t.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()), lr=args.policy_lr)
        if args.autotune:  # 如果有自动更新alpha参数的步骤
            target_entropy = -torch.prod(torch.Tensor(args.action_shape.shape).to(self.device)).item()
            log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = log_alpha.exp().item()
            a_optimizer = torch.optim.Adam([log_alpha], lr=args.q_lr)

    def train(self, transitions):
        r = transitions['r_%d' % self.id]  # 训练时只需要自己的reward
