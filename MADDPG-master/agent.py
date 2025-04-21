import numpy as np
import torch
import os

from matplotlib import pyplot as plt

from maddpg.maddpg import MADDPG
from MASAC.MASAC import MASAC
from collections import deque


class Agent:
    def __init__(self, agent_id, args, policy, algorithm):
        self.args = args  # 这个的成员的幅值在make_env中传递过来了，所以更改make_env的参数即可
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.agent_id = agent_id
        self.policy = policy
        self.algorithm = algorithm
        self.memory = deque(maxlen=10)
        self.lossSafeNet = []
        self.averageLoss = []

    def select_action(self, o, noise_rate, epsilon, global_info=None, train=True):
        if np.random.uniform() < epsilon and (self.algorithm == "MADDPG"
                                              or self.algorithm == "MADDPG_ATT"
                                              or self.algorithm == "MADDPG_RNN"
                                              or self.algorithm == "TD3"):
            u = np.random.uniform(self.args.low_action, self.args.high_action, self.args.action_shape[self.agent_id])
        elif self.algorithm == "MADDPG" or \
                self.algorithm == "MADDPG_ATT" or \
                self.algorithm == "TD3":
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(
                self.device)  # 给第0位添加一维向量，本来o是(18,)，现在是tensor(1,observation shape)
            pi = self.policy.actor_network(inputs).squeeze(
                0)  # 压缩第0位的一维向量,这里policy的输出是tensor(64,action_shape),最后变成(1,action_shape)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            noise = noise_rate * ((self.args.high_action - self.args.low_action) / 2) * np.random.randn(
                *u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, np.float(-(self.args.high_action - self.args.low_action) / 2),
                        np.float((self.args.high_action - self.args.low_action) / 2))
            if self.algorithm == "MADDPG":
                if self.policy.SafeNet and self.policy.evl_with_safeNet or (not train and self.policy.SafeNet):
                    pi = self.policy.actor_network(inputs)
                    u_safe = self.policy.actorSafe(inputs, pi).squeeze(0).cpu().numpy()
                    u = np.clip(u_safe, np.float(-(self.args.high_action - self.args.low_action) / 2),
                                np.float((self.args.high_action - self.args.low_action) / 2))

            # np.clip(a, a_min, a_max, out=None) 限制其值在amin，amax之间
            # TODO:下面这一行去掉了，这是因为采用连续动作的缘故，如果使用0到max这种格式，应该加上下面这一行, MASAC那里同理
            # u += np.float((self.args.high_action-self.args.low_action)/2)     # 要限制其动作在0到high_action内
        elif self.algorithm == "MADDPG_RNN":
            if len(self.memory) == 0 or len(self.memory) != self.memory.maxlen:
                u = np.random.uniform(self.args.low_action, self.args.high_action,
                                      self.args.action_shape[self.agent_id])
                print("empty deque")
                return u.copy()
            inputs = torch.tensor(list(self.memory), dtype=torch.float32).unsqueeze(0).to(self.device)
            # 给第0位添加一维向量，本来o是(10,10)，现在是tensor(1,10,observation shape)
            pi = self.policy.actor_network(inputs)[0].squeeze(
                0)  # 压缩第0位的一维向量,这里policy的输出是tensor(64,action_shape),最后变成(1,action_shape)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            noise = noise_rate * ((self.args.high_action - self.args.low_action) / 2) * np.random.randn(
                *u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, np.float(-(self.args.high_action - self.args.low_action) / 2),
                        np.float((self.args.high_action - self.args.low_action) / 2))
        elif self.algorithm == "IQL":
            input = torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.policy.act(input)
            return action
        elif self.algorithm == "MASAC" or self.algorithm == "MAPPO" or self.algorithm == "MLGA2C":
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(
                0).to(self.device)
            if self.algorithm == "MASAC" or self.algorithm == "MLGA2C":
                pi2, prob, _, _ = self.policy.policy.get_actions(inputs)
            else:
                pi2, prob, _ = self.policy.policy.get_actions(inputs)
            pi = pi2
            pi = pi.squeeze(0)
            u = pi.cpu().numpy()
            u = np.clip(u, np.float(-(self.args.high_action - self.args.low_action) / 2),
                        np.float((self.args.high_action - self.args.low_action) / 2))
            if self.policy.evl_with_safeNet:
                u_safe = self.policy.actorSafe(inputs, pi2).squeeze(0).cpu().numpy()
                u = np.clip(u_safe, np.float(-(self.args.high_action - self.args.low_action) / 2),
                            np.float((self.args.high_action - self.args.low_action) / 2))
            # -----这里下面这一行就是连续动作和离散动作控制不同的点，除此之外还有util部分，需要更改low_action和high_action
            # u += np.int((self.args.high_action - self.args.low_action) / 2)
            # np.clip(a, a_min, a_max, out=None) 限制其值在amin，amax之间
            # u = u * (self.args.high_action-self.args.low_action)/2 + (self.args.high_action+self.args.low_action)/2
            # u = np.clip(u, self.args.low_action, self.args.high_action)
            if self.algorithm == "MAPPO":
                # inputs = torch.cat([inputs, global_info], dim=1)
                # global_info = torch.tensor(global_info).unsqueeze(0)
                if global_info == None:
                    return u.copy()
                global_info = [torch.tensor(row, dtype=torch.float32).unsqueeze(0).to(self.device) for row in
                               global_info]
                values = self.policy.critic(torch.cat(global_info, dim=1))
                return u.copy(), pi2, prob, values
        else:
            print("error arguments input")
            return None
        return u.copy()

    def learn(self, transitions, other_agents, algorithm,
              obs=None,
              next_obs=None,
              values=None, dones=None,
              actions=None, logprobs=None,

              rewards=None, nextdone=None, time_steps=None, safeNet=False, distill=False):
        if algorithm == "MADDPG" or \
                algorithm == "MADDPG_ATT" or \
                algorithm == "MADDPG_RNN" or \
                algorithm == "TD3" \
                or algorithm == "IQL" \
                or self.policy.distill == True:

            if ((
                        algorithm == "MADDPG" or algorithm == "MASAC") and self.policy.SafeNet and time_steps > 120000) or self.policy.distill:
                loss = self.policy.safeNetTrain(transitions).cpu().detach().numpy()
                self.lossSafeNet.append(loss)
                if len(self.lossSafeNet) > 10:
                    self.averageLoss.append(np.mean(self.lossSafeNet[-10:]))

                if time_steps % 100 == 0:
                    np.save(self.policy.model_path + "_loss", np.array(self.lossSafeNet))
                    np.save(self.policy.model_path + "_last_10_mean_loss", np.array(self.averageLoss))
                if time_steps % 1000 == 0:
                    plt.figure()
                    plt.plot(range(len(self.lossSafeNet)), self.lossSafeNet, label='loss')
                    plt.plot(range(len(self.averageLoss)), self.averageLoss, label='last 10 mean loss')
                    plt.xlabel('timestep')
                    plt.ylabel('loss value')
                    plt.legend()
                    plt.savefig(self.policy.model_path + '/safeNet_loss' + '.png', format='png')
                    plt.close()
            else:
                self.policy.train(transitions, other_agents)
        elif algorithm == "MASAC":
            self.policy.train(transitions, self.agent_id)
        elif algorithm == "MLGA2C":
            self.policy.train(transitions, self.agent_id)
        elif algorithm == "MAPPO":
            self.policy.train(obs, next_obs, values, dones, actions, logprobs, rewards, nextdone, self.agent_id,
                              time_steps)
        else:
            print("error arguments input")

    def memory_reset(self):
        self.memory.clear()

    def store_memory(self, obs):
        # 滑动窗口来更新记忆
        self.memory.append(obs)
