from multipledispatch import dispatch
from .Qnet import Qnet
import torch
import os
import numpy as np


class IQL:
    def __init__(self, args, agent_id, iterations, number_actions_per_dim=50, action_dim=2, epsilon=0.1):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        self.Algorithm = "IQL"
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.iterations = iterations  # 代表第几次训练任务
        self.continToDiscrete = number_actions_per_dim # 连续动作空间离散化
        self.action_dim = action_dim
        self.action_space = np.linspace(-1, 1, self.continToDiscrete)  # 离散化动作空间
        # create the network
        self.Q = Qnet(args, self.continToDiscrete**action_dim).to(self.device)
        # using for network architecture
        # self.Wrapper = Wrapper(args, agent_id).to(self.device)
        self.evaluate = self.args.evaluate
        # build up the target network
        self.Q_Target = Qnet(args, self.continToDiscrete**action_dim).to(self.device)
        self.epsilon = epsilon

        # load the weights into the target networks
        self.Q_Target.load_state_dict(self.Q.state_dict())

        # create the optimizer
        self.Q_optim = torch.optim.Adam(self.Q.parameters(), lr=self.args.lr_critic)

        # TensorboardWriter

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # 加载模型
        # TODO: load model !!!!
        id = 1
        if os.path.exists(self.model_path + '/1_Q_params.pkl') and self.evaluate == True:
            self.load_model()

    @dispatch()
    def load_model(self):
        self.Q.load_state_dict(
            torch.load(self.model_path + '/' + str(self.iterations + 1) + '_Q_params.pkl'))
        print('Agent {} successfully loaded Qnetwork: {}'.format(self.agent_id,
                                                                 self.model_path + '/_Q_params.pkl'))

    @dispatch(int)
    def load_model(self, id):
        self.Q.load_state_dict(
            torch.load(self.model_path + '/' + str(id + 1) + '_Q_params.pkl'))
        print('Agent {} successfully loaded Qnetwork: {}'.format(self.agent_id,
                                                          self.model_path + '/_Q_params.pkl'))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.uniform(-1, 1, self.action_dim)  # 随机选择动作
        with torch.no_grad():
            q_values = self.Q(state)
            q_values = q_values.squeeze().cpu().numpy()
            action_idx = np.unravel_index(np.argmax(q_values), (self.continToDiscrete,) * self.action_dim)
            action = [self.action_space[i] for i in action_idx]
        return np.array(action)

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.Q_Target.parameters(), self.Q.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32).to(self.device)
        r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])

        # calculate the target Q value function



        q_next = self.Q_Target(o_next[self.agent_id]).detach()
        max_next_q_values = q_next.max(1)[0]
        target_q = (r.unsqueeze(1) + self.args.gamma * max_next_q_values).detach()

        # the q loss
        q_values = self.Q(o[self.agent_id])
        action_idx = u[self.agent_id]  #已经是连续值了
        # 重新化为离散坐标
        normalized = (action_idx + 1) / 2
        discrete_index = torch.round(normalized * (self.continToDiscrete - 1)).long()
        linear_index = discrete_index[:, 0] * self.continToDiscrete + discrete_index[:, 1]
        q_value = torch.gather(q_values, 1, linear_index.view(-1, 1))  # 按照动作索引选择Q值
        loss = torch.mean((q_value - target_q) ** 2)
        self.Q_optim.zero_grad()
        loss.backward()
        self.Q_optim.step()


        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step, self.iterations)
        self.train_step += 1


    def save_model(self, train_step, iterations):
        # 这里的iterations代表第几次迭代
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        # torch.save(self.critic_network.state_dict(), model_path + '/' + num + '_critic_params.pkl')
        torch.save(self.Q.state_dict(), model_path + '/' + str(iterations) + '_Q_params.pkl')


    def set_mode(self, signal):
        # 将MADDPG设置为评估模式或者是训练模式, 0是评估， 其他是训练
        if signal == 0:
            self.evaluate = True
        else:
            self.evaluate = False