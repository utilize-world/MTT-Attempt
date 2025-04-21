from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
from common.replay_buffer_rnn import Buffer_RNN
from common.replay_buffer_safe import replay_buffer_safe
import torch
import os
import copy as cp
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from AttentionModule import dataloader
# import tensor_b
# using for tensorboard
from common.tensor_b import TensorboardWriter, MTT_tensorboard

from common.utils import check_agent_bound, check_agent_near_bound, ezDrawAPic, is_success, all_ones, \
    is_agents_success_track, is_collision, is_success_col
from maddpg.maddpg import MADDPG
from maddpg.maddpg_rnn import MADDPG_RNN
from maddpg.TD3 import TD3
from MASAC.MASAC import MASAC
from MAPPO.MAPPO import MAPPO
from MLGA2C.MLGA2C import MLGA2C
from MADDPG_ATT.MADDPG_ATT import MADDPG_ATT
from IQL.IQL import IQL
from SafeModule.correct_actions import Correct

class Runner:
    """
    input: (args, env)
    """

    def __init__(self, args, env, algorithm, number, scale=0.1, soft=True, non_filter=False, CBF=False, safeNet=True):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.number = number
        self.algorithm = algorithm
        self.scale = scale # 放缩动作空间，便于在实际场景中动作控制
        self.safeNet = safeNet
        self.soft = soft
        if soft and not non_filter:
            self.buffer = replay_buffer_safe(args)
        else:
            self.buffer = Buffer(args)
        self.dataloader = None
        self.corrector = None # 动作过滤器
        self.soft = soft
        self.CBF = CBF
        # if not non_filter:
        #     if self.soft:
        #         self.algorithm += "_soft"
        #     else:
        #         self.algorithm += "_hard"
        self.non_filter = non_filter
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name + '/' + self.algorithm
        # 规定存放csv的路径位置以及存放fig的位置
        self.csv_save_dir = self.args.csv_save_dir + '/' + self.args.scenario_name + '/' + self.algorithm
        self.fig_save_dir = self.args.fig_save_dir + '/' + self.args.scenario_name + '/' + self.algorithm

        # 保存记录数据
        self.save_data_path = os.path.dirname(
            os.path.abspath(__file__)) + '/data/' + self.args.scenario_name + '/' + self.algorithm + '/'
        if not non_filter:
            if self.soft:
                self.save_path += "_soft"
                self.csv_save_dir += "_soft"
                self.fig_save_dir += "_soft"
                self.save_data_path = self.save_data_path[:-1]
                self.save_data_path += "_soft/"
            elif self.CBF:
                self.save_path += "_CBF"
                self.csv_save_dir += "_CBF"
                self.fig_save_dir += "_CBF"
                self.save_data_path = self.save_data_path[:-1]
                self.save_data_path += "_CBF/"
            else:
                self.save_path += "_hard"
                self.csv_save_dir += "_hard"
                self.fig_save_dir += "_hard"
                self.save_data_path = self.save_data_path[:-1]
                self.save_data_path += "_hard/"
        self.writer = TensorboardWriter(self.args.tensorboard_dir, self.args)
        self.writer.create_writer()
        if not args.writer:
            self.writer.disable_write()  # 不使用writer，加快速度
        self.agents = self._init_agents(self.algorithm)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.csv_save_dir):
            os.makedirs(self.csv_save_dir)
        if not os.path.exists(self.fig_save_dir):
            os.makedirs(self.fig_save_dir)

    def _init_corrector(self):
        abs_path = os.path.dirname(os.path.abspath(__file__))
        abs_path = os.path.dirname(abs_path)
        constraint_dir = abs_path + '/data/' + 'constraint_networks_MADDPG/'
        env_params = self.env.get_env_parameters()
        state_dim = env_params["state_dim"]
        act_dim = env_params["act_dim"]
        N_agents = env_params["num_agents"]
        constraint_dim = env_params["constraint_dim"]
        self.corrector = Correct(N_agents, constraint_dir, state_dim, act_dim, constraint_dim)
        constraint = N_agents * [5 * np.ones(constraint_dim)]
        return constraint

    def _init_agents(self, algorithms):

        agents = []
        if algorithms == "MADDPG":
            for i in range(self.args.n_agents):
                policy = MADDPG(self.args, i, iterations=self.number, writer=self.writer)
                if self.args.evaluate:
                    policy.evl_with_safeNet = True
                # Wrapper = policy.Wrapper
                # self.writer.tensorboard_model_collect(Wrapper, algorithms)

                agent = Agent(i, self.args, policy, algorithms)
                agents.append(agent)
        elif algorithms == "TD3":
            for i in range(self.args.n_agents):
                policy = TD3(self.args, i, iterations=self.number, writer=self.writer)
                Wrapper = policy.Wrapper
                self.writer.tensorboard_model_collect(Wrapper, algorithms)

                agent = Agent(i, self.args, policy, algorithms)
                agents.append(agent)
        elif algorithms == "MADDPG_RNN":
            self.buffer = Buffer_RNN(self.args)
            # 用自己的随机连续抽样
            for i in range(self.args.n_agents):
                policy = MADDPG_RNN(self.args, i, iterations=self.number, writer=self.writer)
                Wrapper = policy.Wrapper
                self.writer.tensorboard_model_collect(Wrapper, algorithms)

                agent = Agent(i, self.args, policy, algorithms)
                agents.append(agent)

        elif algorithms == "MADDPG_ATT":
            policy = MADDPG_ATT(self.args, self.number, self.writer)  # 采用共享参数的方式
            Wrapper = policy.Wrapper
            self.writer.tensorboard_model_collect(Wrapper, algorithms)
            for i in range(self.args.n_agents):
                agent = Agent(i, self.args, policy, algorithms)
                agents.append(agent)
        elif algorithms == "MASAC":

            for i in range(self.args.n_agents):
                policy = MASAC(self.args, i, iterations=self.number)  # 采用共享参数的方式
                agent = Agent(i, self.args, policy, algorithms)
                agents.append(agent)
        elif algorithms == "MLGA2C":
            policy = MLGA2C(self.args, self.writer)  # 采用共享参数的方式
            Wrapper = policy.Wrapper
            self.writer.tensorboard_model_collect(Wrapper, algorithms)
            for i in range(self.args.n_agents):
                agent = Agent(i, self.args, policy, algorithms)
                agents.append(agent)
        elif algorithms == "MAPPO":
            policy = MAPPO(self.args)  # 采用共享参数的方式
            for i in range(self.args.n_agents):
                agent = Agent(i, self.args, policy, algorithms)
                agents.append(agent)
        elif algorithms == "IQL":
            for i in range(self.args.n_agents):
                policy = IQL(self.args, i, iterations=self.number)
                agent = Agent(i, self.args, policy, algorithms)
                agents.append(agent)
        return agents

    def run(self):
        returns = []  # 一次完整测试(多个epi)的平均总回报，主要是训练过程中每过一段时间就会测试模型性能
        success_rate = []  # 真正的成功率，即连续n个时间步，所有target都被观察到，则此episode为成功，否则均失败
        epi_success_rate = []  # epi成功率，即用1个episode中所有target都被观察到时间步 / 总时间
        each_target_sr = []  # 每个target的被追踪率，计算是单个epi每个target被追踪的时间步 / 总时间步
        rewards = 0  # 用来计算单个epi有多少总回报的变量
        train_returns = []  # 来看每个timestep的reward变化
        agent_track_rate = []
        collision_rate = []
        constraint = self._init_corrector()  # 初始化
        done = False
        rewards_list = []  # 用来存储上面的rewards的
        done_list = []
        rewards_epi = [0] * len(self.agents)  # 用来计算每个agent对应的reward_epi,individual，所以有多少个agent就有多少个
        rewards_epi_list = []  # 用来存rewards——epi
        success_epi = 0  # 记录多少episode成功了
        epi_num = 0  # 记录当前epi数
        out_symbol = False
        critical_done = False

        # save data record:
        r_epi_reward = []  # 每个epi 的奖励
        r_success_steps = []  # 每个epi 成功的步数，也就是全部观察到的步数
        r_collisions = []  # 每个epi 发生碰撞的步数
        r_agent_obs_steps = []  # 存储每个epi, 各agent的观察到target的步数
        r_target_obs_steps = []  # 存储每个epi, 各target被观察到的步数

        if self.algorithm == "MAPPO":
            device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
            u_joint = torch.zeros((len(self.agents), self.args.max_episode_len) + (1, self.args.action_shape[0])).to(
                device)
            # 3 * 200 * (1 * 2)
            obs_joint = torch.zeros((len(self.agents), self.args.max_episode_len) + (1, self.args.obs_shape[0])).to(
                device)
            # 3* 200 * 1 *19
            next_obs = torch.zeros((len(self.agents), self.args.max_episode_len) + (1, self.args.obs_shape[0])).to(
                device)
            logprobs = torch.zeros((len(self.agents), self.args.max_episode_len, 1)).to(device)
            # rewards is shared
            rewards_mappo_timestep = torch.zeros((len(self.agents), self.args.max_episode_len, 1)).to(device)
            dones = torch.zeros((len(self.agents), self.args.max_episode_len, 1)).to(device)
            nextdone = torch.zeros((len(self.agents), self.args.max_episode_len, 1)).to(device)
            value = torch.zeros((len(self.agents), self.args.max_episode_len, 1)).to(device)

        s = self.env.reset()
        current_time_step = 0
        cumulate_success_step = 0
        success_step = 0  # 记录每个epi的成功步数
        collision_timesteps = 0  # 记录每个epi的碰撞
        agent_obs_steps = np.zeros(len(self.env.world.agents), dtype=int)  # 记录每个epi各个agent的观察数
        target_obs_steps = np.zeros(len(self.env.world.targets_u), dtype=int)  # 记录每个epi各个target的观察数
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            # debug
            # begin_debug(time_step % 70000 == 0 and time_step > 0)
            # writer可以向MADDPG类传入timestep信息
            self.writer.get_timestep(time_step)
            # if (current_time_step % self.episode_limit == 0) or not (False in done) or critical_done:
            if (current_time_step % self.episode_limit == 0) or critical_done or is_success(cumulate_success_step):
                # 这里就是一个episode结束的位置
                # TODO: EPI 结束位置
                s = self.env.reset()
                for agent in self.agents:
                    agent.memory_reset()
                if self.algorithm == "MAPPO":
                    u_joint.zero_()
                    obs_joint.zero_()
                    logprobs.zero_()
                    value.zero_()
                    rewards_mappo_timestep.zero_()
                    next_obs.zero_()
                    dones.zero_()
                    nextdone.zero_()

                # TODO:数据记录
                if current_time_step != 0:
                    current_time_step = 0
                    r_epi_reward.append(rewards)
                    r_success_steps.append(success_step)
                    r_agent_obs_steps.append(agent_obs_steps)
                    r_collisions.append(collision_timesteps)
                    r_target_obs_steps.append(target_obs_steps)

                rewards_epi_list.append(rewards_epi)  # 这里的形式应该是[[r1t1,r2t1,r3t1..],[r1t2,r1t2]...]
                # 其中ti就代表了第几个episode的reward和。
                rewards_epi = [0] * len(self.agents)
                epi_num += 1
                rewards_list.append(rewards)
                self.writer.tensorboard_scalardata_collect(rewards, epi_num, "episode_")

                rewards_list = rewards_list[-50000:]  # 取最后2000个训练episode

                # epi结束置零部分
                rewards = 0
                success_step = 0
                cumulate_success_step = 0
                collision_timesteps = 0
                agent_obs_steps = np.zeros(len(self.agents), dtype=int)
                target_obs_steps = np.zeros(len(self.env.world.targets_u), dtype=int)  # 记录每个epi各个agent的观察数

            u = []
            u_train = [] #用于 IQL store transitions
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    if self.algorithm == "MAPPO":
                        action, pi2, probs, values = agent.select_action(s[agent_id], self.noise, self.epsilon, s)
                        u_joint[agent_id][current_time_step] = pi2
                        logprobs[agent_id][current_time_step] = probs
                        value[agent_id][current_time_step] = values
                        obs_joint[agent_id][current_time_step] = torch.tensor(s[agent_id], dtype=torch.float32)
                    elif self.algorithm == "IQL":
                        action_ori = agent.select_action(s[agent_id], self.noise, self.epsilon)
                        action = action_ori
                        # 转离散值到连续值，用于step

                    else:
                        action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    u.append(action)
                    if self.algorithm == "IQL":
                        u_train.append(action_ori)

                if not self.non_filter:
                    if self.soft:
                        u, _ = self.corrector.correct_actions_soften(s, u, constraint)

                    elif self.CBF:
                        u = self.corrector.correct_actions_CBF(s, u, constraint)

                    else:
                        u = self.corrector.correct_actions_hard(s, u, constraint)
                actions = u.copy()
                # actions = [self.scale * ele for ele in actions]
                split_actions = np.array_split(actions, len(self.agents))
            # 以下一步其实是补全维度，但是在MTT中没有这样一个需要，因为控制的只有agent而没有target
            # for i in range(self.args.n_agents, self.args.n_players):
            #     actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            if self.algorithm == "MADDPG":
                self.env.shared_reward = False  # 这一步是让每个agent根据自己的奖励训练自己

            # TODO：Step部分
            if not self.non_filter:
                if self.soft:
                    s_next, r, done, info, constraint = self.env.step(split_actions)
                elif self.CBF:
                    env_copy = cp.deepcopy(self.env)
                    s_next_ori, r_ori, done_ori, _, _ = env_copy.step(u)
                else:
                    s_next, r, done, info, constraint = self.env.step(split_actions)
            else:
                s_next, r, done, info, _ = self.env.step(u)

            # 这地方新加的，如果全部出界，则结束当前episode，并给予惩罚
            critical_done = check_agent_bound(self.env.world.agents, self.env.world.bound, 0)
            # TODO: Done check
            done_list.append(done)
            if all_ones(done):
                cumulate_success_step += 1
                success_step += 1
            else:
                cumulate_success_step = 0

            # TODO: 对每个timeStep的数据的处理
            for i, a in enumerate(self.env.world.agents):
                others = self.env.world.agents.copy()
                others.remove(a)
                for other_agent in others:
                    if is_collision(a, other_agent):
                        collision_timesteps += 1
                if a.obs_flag:
                    agent_obs_steps[i] += 1
            for i, target in enumerate(self.env.world.targets_u):
                if target.be_observed:
                    target_obs_steps[i] += 1

            # if check_agent_near_bound(self.env.world.agents, self.env.world.bound, 0):
            #     r = [x - 1 for x in r]
            # if critical_done:
            #     # 如果全出界了，直接每人-100
            #     r = [x - 10 for x in r]
            # 单个的时候就不考虑这个奖励了

            rewards_epi = [x + y for x, y in zip(rewards_epi, r)]  # 将每个agent的reward都单独处理，指叠加，为了计算

            rewards += float(sum(r)) / float(len(self.agents))  # 这里的 reward是一个episode的reward，均值
            train_returns.append(rewards)  # 这里的train——returns是取每个时间步的reward，所以这里看的是timestep尺度的reward变化
            train_returns_clip = train_returns[-1000:]  # 取最后1000个时间步

            # TODO：buffer更新部分和训练
            if self.algorithm == "MAPPO":
                for agent in range(len(self.agents)):
                    dones[agent][current_time_step] = done[agent]
                    nextdone[agent][current_time_step] = done[agent]
                    next_obs[agent][current_time_step] = torch.tensor(s_next[agent], dtype=torch.float32)

                    rewards_mappo_timestep[agent][current_time_step] = r[agent]
                s = s_next
            if self.algorithm == "MADDPG" \
                    or self.algorithm == "MASAC" \
                    or self.algorithm == "MLGA2C" \
                    or self.algorithm == "MADDPG_ATT" \
                    or self.algorithm == "MADDPG_RNN" \
                    or self.algorithm == "TD3" \
                    or self.algorithm == "IQL":
                if not self.non_filter:
                    if self.soft:
                        self.buffer.store_episode_safe(s[:self.args.n_agents], u, r[:self.args.n_agents],
                                                  s_next[:self.args.n_agents], split_actions)
                    else:
                        self.buffer.store_episode(s[:self.args.n_agents], split_actions, r[:self.args.n_agents],
                                              s_next[:self.args.n_agents])


                    # if self.CBF:
                    #     self.buffer.store_episode(s[:self.args.n_agents], u, r_ori[:self.args.n_agents],
                    #                               s_next_ori[:self.args.n_agents])
                elif self.algorithm == "IQL":

                    self.buffer.store_episode(s[:self.args.n_agents], u_train, r[:self.args.n_agents],
                                              s_next[:self.args.n_agents])
                else:
                    self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents],
                                          s_next[:self.args.n_agents])
                if False:
                    for i, agent in enumerate(self.agents):
                        agent.store_memory(s[i])
                s = s_next
                if self.buffer.current_size >= self.args.batch_size:
                    transitions = self.buffer.sample(self.args.batch_size)
                    import copy
                    trans = [copy.copy(transitions) for _ in range(len(self.agents))]
                    # 复制多份，因为经过learn之后的critic网络后，transition的动作发生了改变？是什么将u和transition绑定在一起了？
                    for i, agent in enumerate(self.agents):
                        other_agents = self.agents.copy()
                        other_agents.remove(agent)
                        # agent.learn(transitions, other_agents, self.algorithm)
                        if self.safeNet:
                            agent.learn(trans[i], other_agents, self.algorithm, safeNet=True, time_steps=time_step)
                        else:
                            agent.learn(trans[i], other_agents, self.algorithm)
            elif self.algorithm == "MAPPO":
                if (current_time_step > 0 and current_time_step % self.episode_limit == 0) or critical_done:

                    for i, agent in enumerate(self.agents):
                        agent.learn(transitions=None, other_agents=None, algorithm="MAPPO",
                                    obs=obs_joint[:, :current_time_step, :, :],
                                    next_obs=next_obs[:, :current_time_step, :, :],
                                    values=value[i, :current_time_step, :],
                                    dones=dones[i, :current_time_step, :], actions=u_joint[i, :current_time_step, :, :],
                                    logprobs=logprobs[i, :current_time_step, :],
                                    rewards=rewards_mappo_timestep[i, :current_time_step, :],
                                    nextdone=nextdone[i, :current_time_step, :], time_steps=current_time_step)

            current_time_step += 1
            # TODO：画图部分，其实这里有待考究，因为每个episode的时间步都已经不一样了，按道理每次画图检查是根据episode数来
            #  计算，而不是根据timestep数来计算，即evaluate_rate(这个值实际上是时间步有关的值）
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                output = self.evaluate()
                returns.append(output[0])
                success_rate.append(output[1])
                epi_success_rate.append(output[2])
                each_target_sr.append(output[3])
                agent_track_rate.append(output[4])
                collision_rate.append(output[5])

                plt.figure(figsize=(20, 20))

                plt.subplot(2, 2, 1)
                plt.plot(range(len(train_returns_clip)), train_returns_clip)
                plt.xlabel('time steps ')
                plt.ylabel('instant training reward')
                plt.title('training reward(average)')

                plt.subplot(2, 2, 2)
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episode')
                plt.ylabel('average returns')
                plt.title('evaluating average returns')

                plt.subplot(2, 2, 3)
                plt.plot(range(len(rewards_list)), rewards_list)
                plt.xlabel('training episode')
                plt.ylabel('average epi reward')
                plt.title('training episode average rewards')

                plt.subplot(2, 2, 4)
                plt.plot(range(len(success_rate)), success_rate, label='true_success_rate', color='blue', linestyle='-',
                         marker='^')
                plt.plot(range(len(agent_track_rate)), agent_track_rate, label='ave_agent_track_rate', linestyle='-',
                         marker='^')
                plt.plot(range(len(epi_success_rate)), epi_success_rate, label='epi_per_success_rate', color='red',
                         linestyle='-', marker='^')
                plt.plot(range(len(collision_rate)), collision_rate, label='ave_collision_rate', linestyle='-',
                         marker='s')
                plt.legend()
                plt.xlabel('epi')
                plt.ylabel('success_rate or collision rate')
                plt.title('success_rate and collision rate in training')

                plt.savefig(self.save_path + '/plt' + str(self.number) + '.png', format='png')

                plt.close()  # 不用每次都跳出来

                plt.figure()
                for i in range(len(self.env.world.targets_u)):
                    name = 'target' + str(i)
                    plt.plot(range(len(each_target_sr)), np.array(each_target_sr)[:, i],
                             label=name)
                plt.legend()
                plt.xlabel('epi')
                plt.ylabel('target_tracked_rate')
                plt.title('target_tracked_rate in training')
                plt.savefig(self.save_path + '/targetTrackedRate' + str(self.number) + '.png', format='png')
                plt.close()

                # 这里分别画出每个agent独自的奖励(单个epi的奖励和)
                # 本来可以这样做，但是为了方便，用了utils自己的简单的画图函数
                # fig, axs = plt.subplots(2, 1)
                #
                # axs[0].plot(x, y1)
                #
                # axs[1].plot(x, y2)
                for i in range(len(self.agents)):
                    save_path = self.save_path + '/individual_returns(per_epi)' + '/' + str(
                        self.number) + "/agent" + str(i)
                    x_data = range(len(rewards_epi_list))
                    y_data = [rew[i] for rew in rewards_epi_list]  # 相当于取列向量了，一共有epi*agent数的值
                    ezDrawAPic(save_path, f"agent{i}_episode_reward_sum", "episodes", "returns", x_data, y_data)

            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.00005)
            np.save(self.save_path + '/returns.pkl', returns)
        # TODO:整个进程的结束部分，需要进行数据记录
        # ---Path: MADDPG-MPE-copy/data/scenario_name/Algorithms_name/...
        if not os.path.exists(self.save_data_path):
            os.makedirs(self.save_data_path)
        np.save(self.save_data_path + "Epi_Reward", np.array(r_epi_reward))
        np.save(self.save_data_path + "Epi_Collisions", np.array(r_collisions))
        np.save(self.save_data_path + "Epi_Agent_obs_steps", np.array(r_agent_obs_steps))
        np.save(self.save_data_path + "Epi_Target_beObserved_steps", np.array(r_target_obs_steps))
        np.save(self.save_data_path + "Epi_Success_Steps", np.array(r_success_steps))
        np.save(self.save_data_path + "Epi_Success_Rate", np.array(success_rate))




        self.writer.close_writer()
        rewards_epi = rewards_list[1:]
        return rewards_epi

    def evaluate(self):
        returns = []
        epi_success_rate = []
        epi_agent_track_rate = []
        epi_collision_rate = []
        constraint = self._init_corrector()  # 初始化
        each_target_tracked_success_rate = []
        self.env.world.train = False  # 将模式调整为执行
        success_epi = 0
        epi_collision = []
        epi_reward = []


        if self.args.evaluate:
            self.agents = self._init_agents(self.algorithm)
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            successive_success_timestep = 0
            suc = False
            success_timestep = 0
            collision_timesteps = 0
            agent_track_timesteps = 0
            each_target_tracked_timestep = [0] * len(self.env.world.targets_u)

            for time_step in range(self.args.evaluate_episode_len):
                self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0, train=self.env.world.train)
                        actions.append(action)
                # for i in range(self.args.n_agents, self.args.n_players):      # 这一步可能不太需要，所以注释
                #     actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

                if not self.non_filter and not self.safeNet:
                    if self.soft:
                        actions, _ = self.corrector.correct_actions_soften(s, actions, constraint)

                    else:
                        actions = self.corrector.correct_actions_hard(s, actions, constraint)
                    # actions = [self.scale * ele for ele in actions]
                split_actions = np.array_split(actions, len(self.agents))
                if not self.non_filter and not self.safeNet:
                    s_next, r, done, info, constraint = self.env.step(split_actions)
                else:
                    s_next, r, done, info, _ = self.env.step(actions)
                # critical_done = check_agent_bound(self.env.world.agents, self.env.world.bound, 0)
                dis_map = self.env.world.distance_cal_target()
                self.env.world.distance_cal_agent()

                for a in self.env.world.agents:
                    others = self.env.world.agents.copy()
                    others.remove(a)
                    for other_agent in others:
                        if is_collision(a, other_agent):

                            collision_timesteps += 1
                if is_agents_success_track(dis_map, self.env.world.agents[0].obs_range):
                    agent_track_timesteps += 1

                for i, target in enumerate(self.env.world.targets_u):
                    if target.be_observed:
                        each_target_tracked_timestep[i] += 1

                if all_ones(done):
                    print("successive count")
                    successive_success_timestep += 1
                    success_timestep += 1

                else:
                    successive_success_timestep = 0
                # rewards += r[0]     # ？？为什么是r[0],是因为采用shared reward？
                rewards += sum(r) / len(self.agents)
                if is_success_col(successive_success_timestep, collision_timesteps):
                    suc = True

                s = s_next
            epi_collision.append(collision_timesteps)

            collision_rate = float(collision_timesteps) / self.args.evaluate_episode_len
            agent_track_rate = float(agent_track_timesteps) / self.args.evaluate_episode_len
            per_episode_success_rate = success_timestep / self.args.evaluate_episode_len
            each_target_tracked_success_rate_single = [float(i) / self.args.evaluate_episode_len for i in
                                                       each_target_tracked_timestep]

            epi_agent_track_rate.append(agent_track_rate)
            epi_collision_rate.append(collision_rate)
            epi_success_rate.append(per_episode_success_rate)
            each_target_tracked_success_rate.append(each_target_tracked_success_rate_single)
            returns.append(rewards)

            if suc and collision_timesteps == 0:
                success_epi += 1
                print(f"success_epi = {success_epi}")
            print('Returns is', rewards)
        if self.args.evaluate:
            np.save(self.save_data_path + "Test_Epi_Reward", np.array(returns))
            np.save(self.save_data_path + "Test_Epi_collisions", np.array(epi_collision))
            np.save(self.save_data_path + "Test_Epi_collisionRate", np.array(epi_collision_rate))

        plt.figure()
        plt.plot(range(len(epi_success_rate)), epi_success_rate, label='both_obs')
        for i, target in enumerate(self.env.world.targets_u):
            # 记录每个target的单个epi追踪成功率
            name = 'target' + str(i)
            plt.plot(range(len(each_target_tracked_success_rate)), np.array(each_target_tracked_success_rate)[:, i],
                     label=name)

        plt.legend()
        plt.xlabel('epi')
        plt.ylabel('epi_success_rate')
        plt.title('success_rate in Exec')
        plt.savefig(self.save_path + '/TestEpiSuccessRate' + str(self.number) + '.png', format='png')
        plt.close()

        print("The success rate = ", float(success_epi / self.args.evaluate_episodes))  # 计算成功率
        print("The ave_epi success rate = ", np.mean(epi_success_rate))  # 计算成功率
        print("The ave_epi target tracking success rate = ", np.mean(each_target_tracked_success_rate, axis=0))  # 计算成功率

        return sum(returns) / self.args.evaluate_episodes, \
               float(success_epi / self.args.evaluate_episodes), \
               np.mean(epi_success_rate), \
               np.mean(each_target_tracked_success_rate, axis=0), \
               np.mean(epi_agent_track_rate), \
               np.mean(epi_collision_rate)

    # def _train_step(self, batch):
    #     for key in batch.keys():
    #         batch[key] = batch[key].to(self.device)
    #
    #     for i, agent in enumerate(self.agents):
    #         other_agents = self.agents.copy()
    #         other_agents.remove(agent)
    #         agent.learn(batch, other_agents, self.algorithm)
