from multipledispatch import dispatch

import torch
import os
from .actor_critic import Actor, Critic, Wrapper


class SafeMADDPG:
    def __init__(self, args, agent_id, iterations, writer):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        self.Algorithm = "SafeMADDPG"
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.iterations = iterations  # 代表第几次训练任务
        # create the network
        self.actor_network = Actor(args, agent_id).to(self.device)
        self.critic_network = Critic(args).to(self.device)
        # using for network architecture
        self.Wrapper = Wrapper(args, agent_id).to(self.device)
        self.evaluate = self.args.evaluate
        # build up the target network
        self.actor_target_network = Actor(args, agent_id).to(self.device)
        self.critic_target_network = Critic(args).to(self.device)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # TensorboardWriter
        self.writer = writer

        # SafeModule params
        self.solver_interventions = 0
        self.solver_infeasible = 0



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
        if os.path.exists(self.model_path + '/1_actor_params.pkl') and self.evaluate == True:
            self.load_model()

    @dispatch()
    def load_model(self):
        self.actor_network.load_state_dict(
            torch.load(self.model_path + '/' + str(self.iterations + 1) + '_actor_params.pkl'))
        self.critic_network.load_state_dict(
            torch.load(self.model_path + '/' + str(self.iterations + 1) + '_critic_params.pkl'))
        print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                      self.model_path + '/_actor_params.pkl'))
        print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                       self.model_path + '/_critic_params.pkl'))

    @dispatch(int)
    def load_model(self, id):
        self.actor_network.load_state_dict(
            torch.load(self.model_path + '/' + str(id + 1) + '_actor_params.pkl'))
        self.critic_network.load_state_dict(
            torch.load(self.model_path + '/' + str(id + 1) + '_critic_params.pkl'))
        print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                      self.model_path + '/' + str(
                                                                          id + 1) + '/_actor_params.pkl'))
        print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                       self.model_path + '/' + str(
                                                                           id + 1) + '/_critic_params.pkl'))

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
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
        u_next = []

        with torch.no_grad():
            # 得到下一个状态对应的动作
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_network(o_next[agent_id]))
                else:
                    # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
                    u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
                    index += 1
            q_next = self.critic_target_network(o_next, u_next).detach()

            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

            # the q loss
        q_value = self.critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        u[self.agent_id] = self.actor_network(o[self.agent_id])

        actor_loss = - self.critic_network(o, u).mean()
        # if self.agent_id == 0:
        #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # update the network
        # TODO: Test, recording actorloss and critic loss
        # print(r.device)  # 确认奖励张量在GPU上
        # print(q_value.device)  # 确认Q值计算在GPU上
        # print(actor_loss.device)  # 确认Actor损失在GPU上

        self.writer.tensorboard_scalardata_collect(actor_loss, self.writer.time_step, f"actor_loss_{self.agent_id}_")
        self.writer.tensorboard_scalardata_collect(critic_loss, self.writer.time_step, f"critic_loss_{self.agent_id}_")
        self.actor_optim.zero_grad()  # 每次更新前，必须将所要更新梯度的网络梯度置0，因为梯度是累积的
        # 在通过backward计算完梯度后(产生梯度)，经过step来通过梯度下降法更新参数的值
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        # TODO:记录参数变化
        self.writer.tensorboard_histogram_collect(self.actor_network, self.critic_network, self.writer.time_step,
                                                  self.agent_id)

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
        torch.save(self.actor_network.state_dict(), model_path + '/' + str(iterations) + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(), model_path + '/' + str(iterations) + '_critic_params.pkl')

    def set_mode(self, signal):
        # 将MADDPG设置为评估模式或者是训练模式, 0是评估， 其他是训练
        if signal == 0:
            self.evaluate = True
        else:
            self.evaluate = False

