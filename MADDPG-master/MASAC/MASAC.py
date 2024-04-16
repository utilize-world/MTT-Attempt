import torch
import os
from .actor_critic import Actor, Q_net
import einops
import torch.nn.functional as F


class MASAC:
    def __init__(self, args):
        self.args = args
        self.train_step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.policy = Actor(args, 0)
        # 创建单个policy_net,这里的0，就是代表第0个agent的actor，由于单个actor，
        # 在homogeneous情况下随机一个都行
        self.q_lr = 1e-3

        self.qf1 = Q_net(args).to(self.device)
        self.qf2 = Q_net(args).to(self.device)

        self.qf1_t = Q_net(args)
        self.qf2_t = Q_net(args)

        self.qf1_t.load_state_dict(self.qf1.state_dict())
        self.qf2_t.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = torch.optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.q_lr)

        self.p_lr = 3e-4
        self.actor = Actor(args, 0).to(self.device)
        # self.actor_t = Actor(args, agent_id)
        self.alpha = args.alpha
        # self.actor_t.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()), lr=self.p_lr)
        if args.autotune:  # 如果有自动更新alpha参数的步骤
            target_entropy = -torch.prod(torch.Tensor(args.action_shape.shape).to(self.device)).item()
            log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = log_alpha.exp().item()
            a_optimizer = torch.optim.Adam([log_alpha], lr=args.q_lr)

    # 先留着用来加载和存放model参数

    def save_model(self, train_step, agent_id):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.policy.state_dict(), model_path + '/' + 'MASAC' + '/' + num + 'MASAC_actor_params.pkl')
        torch.save(self.qf1.state_dict(), model_path + '/' + 'MASAC' + '/' + num + 'MASAC_critic_q1_params.pkl')
        torch.save(self.qf2.state_dict(), model_path + '/' + 'MASAC' + '/' + num + 'MASAC_critic_q2_params.pkl')


    def load_model(self):
        pass

    def _soft_update_target_network(self):
        for target_param, param in zip(self.qf1.parameters(), self.qf1_t.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.qf2.parameters(), self.qf2_t.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    def train(self, transitions, agent_id):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)

        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
        r = transitions['r_%d' % agent_id]  # 训练时只需要自己的reward
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])  # 大小是agent_number*batch*observation_shape
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])
        u_next = []
        next_state_log = []
        with torch.no_grad():

            #u_next, next_state_log_pi = self.policy.get_actions(o_next)#报错，格式不对
            for i in range(self.args.n_agents):
                u_next_i, next_state_log_i, _ = self.policy.get_actions(o_next[i])
                u_next.append(u_next_i)
                next_state_log.append(next_state_log_i)
            # u_next: agent_number * batch_size
            # q_next: agent_number * 1
            q1_next_target = self.qf1_t(o_next, u_next).detach()
            q2_next_target = self.qf2_t(o_next, u_next).detach()
            next_state_log_pi = einops.reduce(
                next_state_log, "c b a -> () b a", "sum"
            )
            # 这里相当于把每个agent的log概率加起来，结构应该是这样的
            #                  表示的只是数据的数组形式，并不是一种数据结构
            #   agent\ batch内的sample，logΠ      1       2       3                            1
            #           1                                                          1      prob_sum1
            #           2                                                   ---->  2      prob_sum2
            #           3                                                          3      prob_sum3
            # SAC bellman equation,注意这里的输入是observation的concatenate, 而没有加入global information,这点之后再考虑

            qf1_a_values = self.qf1(o, u).view(-1)
            qf2_a_values = self.qf2(o, u).view(-1)
            min_qf_next_target = (torch.min(q1_next_target,
                                           q2_next_target) - self.alpha * next_state_log_pi) # 相当于往Q里加了log
            next_q_value = r.unsqueeze(1) + self.args.gamma * (min_qf_next_target).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf1_loss = (next_q_value - qf1_a_values).pow(2).mean()
            qf2_loss = (next_q_value - qf2_a_values).pow(2).mean()
            qf_loss = (qf1_loss + qf2_loss).requires_grad_(True)

            self.q_optimizer.zero_grad()
            qf_loss.backward()
            self.q_optimizer.step()

            # actor update,之前的数据都是在联合角度来看，actor的更新是对于每个agent本身的本地观察生效，而且遵循TD3的延迟更新
            if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
                # 如果确认更新
                u_pi = []
                state_log_pi = []
                for i in range(self.args.n_agents):
                    u_pi_i, state_log_pi_i = self.policy.get_actions(o[i])
                    u_pi.append(u_pi_i)
                    state_log_pi.append(state_log_pi_i)
                log_pi = einops.reduce(
                state_log_pi, "c b a -> () b a", "sum"
            )
                qf1_pi = self.qf1(o, u_pi)
                qf2_pi = self.qf2(o, u_pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
                    self.save_model(self.train_step, agent_id)
                self.train_step += 1

