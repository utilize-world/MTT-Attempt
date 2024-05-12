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
        # self.alpha = args.alpha
        ## auto
        self.target_entropy = -torch.prod(torch.Tensor([1, self.args.action_shape[0]]).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=self.q_lr)

        # self.actor_t.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()), lr=self.p_lr)
        # if args.autotune:  # 如果有自动更新alpha参数的步骤
        #     target_entropy = -torch.prod(torch.Tensor(args.action_shape.shape).to(self.device)).item()
        #     log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        #     self.alpha = log_alpha.exp().item()
        #     a_optimizer = torch.optim.Adam([log_alpha], lr=args.q_lr)

    # 先留着用来加载和存放model参数

    def save_model(self, train_step, agent_id):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        model_path = os.path.join(model_path, 'MASAC')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.policy.state_dict(), model_path + '/' + num + 'MASAC_actor_params.pkl')
        torch.save(self.qf1.state_dict(), model_path + '/' + num + 'MASAC_critic_q1_params.pkl')
        torch.save(self.qf2.state_dict(), model_path + '/' + num + 'MASAC_critic_q2_params.pkl')

    def load_model(self):
        pass

    def _soft_update_target_network(self):
        for param, target_param in zip(self.qf1.parameters(), self.qf1_t.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for param, target_param in zip(self.qf2.parameters(), self.qf2_t.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    def train(self, transitions, agent_id):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        # transition实际上是一个字典，是agent数量*4个key，每个key对应的是tensor(batchsize,space)    如o_0 : tensor(256*19)
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
        r = transitions['r_%d' % agent_id]  # 训练时只需要自己的reward
        # r: tensor(256, )
        # o,u,onext : list:2, tensor(256,19), tensor(256,19), for u, list2, tensor(256, 2) tensor(256, 2)
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])  # 大小是agent_number*batch*observation_shape
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])
            #   结构与之前基本一致，只不过是根据id进行重新分开
            #
        u_next = []
        next_state_log = []
        with torch.no_grad():

            # u_next, next_state_log_pi = self.policy.get_actions(o_next)#报错，格式不对
            for i in range(self.args.n_agents):
                u_next_i, next_state_log_i, _ = self.policy.get_actions(o_next[i])
                # 256*2, 256*1
                u_next.append(u_next_i)  # list2, tensor 256*2
                next_state_log.append(next_state_log_i)  # list2,tensor 256*1
            # next_state_log : 2 * tensor(256,1)
            # u_next: agent_number * batch_size * action shape.  e.g. 2* tensor(256, 2)
            # q_next: agent_number * batch * 1          e.g. 2 * tensor(256, 1)
            q1_next_target = self.qf1_t(o_next, u_next)
            q2_next_target = self.qf2_t(o_next, u_next)
            next_state_log_pi = einops.reduce(
                next_state_log, "c b a -> b a", "sum"
            )
            min_qf_next_target = (torch.min(q1_next_target,
                                            q2_next_target) - self.alpha * next_state_log_pi)  # 相当于往Q里加了log
            next_q_value = (r.unsqueeze(1) + self.args.gamma * (min_qf_next_target)).view(-1)
            # 这里相当于把每个agent的log概率加起来，结构应该是这样的
            #
            #                  表示的只是数据的数组形式，并不是一种数据结构
            #   agent\ batch内的sample，logΠ      1       2       3                            1
            #           1                                                          1      prob_sum1
            #           2                                                   ---->  2      prob_sum2
            #           3                                                          3      prob_sum3
            # SAC bellman equation,注意这里的输入是observation的concatenate, 而没有加入global information,这点之后再考虑

        qf1_a_values = self.qf1(o, u).view(-1)
        qf2_a_values = self.qf2(o, u).view(-1)
        # qf1_a_value tensor (256,)

        # 这里应该有一个（1-done）在gamma后才对
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        # qf1_loss = (next_q_value - qf1_a_values).pow(2).mean()
        # qf2_loss = (next_q_value - qf2_a_values).pow(2).mean()
        qf_loss = (qf1_loss + qf2_loss)
        # qf_loss = (qf1_loss + qf2_loss)
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()
        self.train_step += 1
        # actor update,之前的数据都是在联合角度来看，actor的更新是对于每个agent本身的本地观察生效，而且遵循TD3的延迟更新
        # if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
        if self.train_step > 0 and self.train_step % 1 == 0:
            # 如果确认更新
            u_pi = []
            state_log_pi = []
            for i in range(self.args.n_agents):
                u_pi_i, state_log_pi_i, _ = self.policy.get_actions(o[i])
                u_pi.append(u_pi_i)
                state_log_pi.append(state_log_pi_i)
            log_pi = einops.reduce(
                state_log_pi, "c b a -> b a", "sum"
            )
            qf1_pi = self.qf1(o, u_pi)
            qf2_pi = self.qf2(o, u_pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
            actor_loss = (((self.alpha * log_pi) - min_qf_pi).mean())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            ## autotuned alpha
            with torch.no_grad():
                for i in range(self.args.n_agents):
                    u_pi_i, state_log_pi_i, _ = self.policy.get_actions(o[i])
                    u_pi.append(u_pi_i)
                    state_log_pi.append(state_log_pi_i)
                log_pi = einops.reduce(
                    state_log_pi, "c b a -> b a", "sum"
                )
            alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            ###
        self._soft_update_target_network()

        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step, agent_id)
