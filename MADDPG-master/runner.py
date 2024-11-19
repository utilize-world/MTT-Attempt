from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
from common.replay_buffer_rnn import Buffer_RNN
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from AttentionModule import dataloader
import tensor_b
# using for tensorboard
from tensor_b import TensorboardWriter, MTT_tensorboard

from common.utils import check_agent_bound, check_agent_near_bound, ezDrawAPic, is_success, all_ones, is_agents_success_track, is_collision
from maddpg.maddpg import MADDPG
from maddpg.maddpg_rnn import MADDPG_RNN
from maddpg.TD3 import TD3
from MASAC.MASAC import MASAC
from MAPPO.MAPPO import MAPPO
from MLGA2C.MLGA2C import MLGA2C
from MADDPG_ATT.MADDPG_ATT import MADDPG_ATT

class Runner:
    """
    input: (args, env)
    """

    def __init__(self, args, env, algorithm, number):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.number = number
        self.algorithm = algorithm

        self.buffer = Buffer(args)
        self.dataloader = None
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name + '/' + self.algorithm
        ## 规定存放csv的路径位置以及存放fig的位置
        self.csv_save_dir = self.args.csv_save_dir + '/' + self.args.scenario_name + '/' + self.algorithm
        self.fig_save_dir = self.args.fig_save_dir + '/' + self.args.scenario_name + '/' + self.algorithm

        self.writer = TensorboardWriter(self.args.tensorboard_dir, self.args)
        self.writer.create_writer()
        if args.writer==False:
            self.writer.disable_write() # 不使用writer，加快速度
        self.agents = self._init_agents(self.algorithm)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.csv_save_dir):
            os.makedirs(self.csv_save_dir)
        if not os.path.exists(self.fig_save_dir):
            os.makedirs(self.fig_save_dir)

    def _init_agents(self, algorithms):

        agents = []
        if algorithms == "MADDPG":
            for i in range(self.args.n_agents):
                policy = MADDPG(self.args, i, iterations=self.number, writer=self.writer)
                Wrapper = policy.Wrapper
                self.writer.tensorboard_model_collect(Wrapper, algorithms)

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
            policy = MASAC(self.args)  # 采用共享参数的方式
            for i in range(self.args.n_agents):
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
        return agents

    def run(self):
        returns = []    # 一次完整测试(多个epi)的平均总回报，主要是训练过程中每过一段时间就会测试模型性能
        success_rate = []   # 真正的成功率，即连续n个时间步，所有target都被观察到，则此episode为成功，否则均失败
        epi_success_rate =[]    # epi成功率，即用1个episode中所有target都被观察到时间步 / 总时间
        each_target_sr = []     # 每个target的被追踪率，计算是单个epi每个target被追踪的时间步 / 总时间步
        rewards = 0         # 用来计算单个epi有多少总回报的变量
        train_returns = []  # 来看每个timestep的reward变化
        agent_track_rate = []
        collision_rate = []

        done = False
        rewards_list = []   # 用来存储上面的rewards的
        done_list = []
        rewards_epi = [0] * len(self.agents)  # 用来计算每个agent对应的reward_epi,individual，所以有多少个agent就有多少个
        rewards_epi_list = []   # 用来存rewards——epi
        success_epi = 0 # 记录多少episode成功了
        epi_num = 0 # 记录当前epi数
        out_symbol = False
        critical_done = False


        if self.algorithm == "MAPPO":
            device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
            u_joint = torch.zeros((len(self.agents), self.args.max_episode_len) + (1, self.args.action_shape[0])).to(
                device)
            # 3 * 200 * (1 * 2)
            obs_joint = torch.zeros((len(self.agents), self.args.max_episode_len) + ((1, self.args.obs_shape[0]))).to(
                device)
            # 3* 200 * 1 *19
            next_obs = torch.zeros((len(self.agents), self.args.max_episode_len) + ((1, self.args.obs_shape[0]))).to(
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
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            # debug
            # begin_debug(time_step % 70000 == 0 and time_step > 0)
            # writer可以向MADDPG类传入timestep信息
            self.writer.get_timestep(time_step)
            # if (current_time_step % self.episode_limit == 0) or not (False in done) or critical_done:
            if (current_time_step % self.episode_limit == 0) or critical_done or is_success(cumulate_success_step):
                # 这里就是一个episode结束的位置
                # TODO: epi data processing
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

                current_time_step = 0

                rewards_epi_list.append(rewards_epi)  # 这里的形式应该是[[r1t1,r2t1,r3t1..],[r1t2,r1t2]...]
                # 其中ti就代表了第几个episode的reward和。
                rewards_epi = [0] * len(self.agents)
                epi_num += 1
                rewards_list.append(rewards)
                # TODO:collect epi reward
                self.writer.tensorboard_scalardata_collect(rewards, epi_num, "episode_")

                rewards_list = rewards_list[-50000:]  # 取最后2000个训练episode
                rewards = 0
                cumulate_success_step = 0

            u = []
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    if self.algorithm == "MAPPO":
                        action, pi2, probs, values = agent.select_action(s[agent_id], self.noise, self.epsilon, s)
                        u_joint[agent_id][current_time_step] = pi2
                        logprobs[agent_id][current_time_step] = probs
                        value[agent_id][current_time_step] = values
                        obs_joint[agent_id][current_time_step] = torch.tensor(s[agent_id], dtype=torch.float32)

                    else:
                        action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    u.append(action)
                    actions.append(action)
            # 以下一步其实是补全维度，但是在MTT中没有这样一个需要，因为控制的只有agent而没有target
            # for i in range(self.args.n_agents, self.args.n_players):
            #     actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            if self.algorithm == "MADDPG":
                self.env.shared_reward = False  # 这一步是让每个agent根据自己的奖励训练自己
            s_next, r, done, info = self.env.step(actions)

            # 这地方新加的，如果全部出界，则结束当前episode，并给予惩罚
            critical_done = check_agent_bound(self.env.world.agents, self.env.world.bound, 0)
            # TODO: Done check
            done_list.append(done)
            if all_ones(done):
                cumulate_success_step += 1
            else:
                cumulate_success_step = 0
            # if check_agent_near_bound(self.env.world.agents, self.env.world.bound, 0):
            #     r = [x - 1 for x in r]
            # if critical_done:
            #     # 如果全出界了，直接每人-100
            #     r = [x - 10 for x in r]
            ################### 单个的时候就不考虑这个奖励了
            # TODO: 这个位置实际就是对每个timestep进行的数据处理

            rewards_epi = [x + y for x, y in zip(rewards_epi, r)]  # 将每个agent的reward都单独处理，指叠加，为了计算

            rewards += float(sum(r)) / float(len(self.agents))  # 这里的 reward是一个episode的reward，均值
            train_returns.append(rewards)  # 这里的train——returns是取每个时间步的reward，所以这里看的是timestep尺度的reward变化
            train_returns_clip = train_returns[-1000:]  # 取最后1000个时间步

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
                    or self.algorithm == "MADDPG_ATT"\
                    or self.algorithm == "MADDPG_RNN"\
                    or self.algorithm == "TD3":

                self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents],
                                          s_next[:self.args.n_agents])
                for i, agent in enumerate(self.agents):
                    agent.store_memory(s[i])
                s = s_next
                if self.buffer.current_size >= self.args.batch_size:
                    transitions = self.buffer.sample(self.args.batch_size)
                    import copy
                    trans = [copy.copy(transitions) for x in range(len(self.agents))]
                    # 复制多份，因为经过learn之后的critic网络后，transition的动作发生了改变？是什么将u和transition绑定在一起了？
                    for i, agent in enumerate(self.agents):
                        other_agents = self.agents.copy()
                        other_agents.remove(agent)
                        # agent.learn(transitions, other_agents, self.algorithm)
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
                plt.plot(range(len(success_rate)), success_rate, label='true_success_rate', color='blue', linestyle='-', marker='^')
                plt.plot(range(len(agent_track_rate)), agent_track_rate, label='ave_agent_track_rate', linestyle='-', marker='^')
                plt.plot(range(len(epi_success_rate)), epi_success_rate, label='epi_per_success_rate', color='red', linestyle='-', marker='^')
                plt.plot(range(len(collision_rate)), collision_rate, label='ave_collision_rate', linestyle='-', marker='s')
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
                    y_data = [rew[i] for rew in rewards_epi_list]   # 相当于取列向量了，一共有epi*agent数的值
                    ezDrawAPic(save_path, f"agent{i}_episode_reward_sum", "episodes", "returns", x_data, y_data)

            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.00005)
            np.save(self.save_path + '/returns.pkl', returns)
        self.writer.close_writer()
        rewards_epi = rewards_list[1:]
        return rewards_epi

    def evaluate(self):
        returns = []
        epi_success_rate = []
        epi_agent_track_rate = []
        epi_collision_rate = []
        each_target_tracked_success_rate = []
        self.env.world.train = False  # 将模式调整为执行
        success_epi = 0
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
            each_target_tracked_timestep = [0]*len(self.env.world.targets_u)
            for time_step in range(self.args.evaluate_episode_len):
                self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                # for i in range(self.args.n_agents, self.args.n_players):      # 这一步可能不太需要，所以注释
                #     actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)
                critical_done = check_agent_bound(self.env.world.agents, self.env.world.bound, 0)
                dis_map = self.env.world.distance_cal_target()
                uav_map = self.env.world.distance_cal_agent()

                if is_collision(uav_map, self.env.world.agents[0].safe_range):
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
                if is_success(successive_success_timestep):
                    suc = True

                s = s_next
            collision_rate = float(collision_timesteps) / self.args.evaluate_episode_len
            agent_track_rate = float(agent_track_timesteps) / self.args.evaluate_episode_len
            per_episode_success_rate = success_timestep / self.args.evaluate_episode_len
            each_target_tracked_success_rate_single = [float(i) / self.args.evaluate_episode_len for i in each_target_tracked_timestep]

            epi_agent_track_rate.append(agent_track_rate)
            epi_collision_rate.append(collision_rate)
            epi_success_rate.append(per_episode_success_rate)
            each_target_tracked_success_rate.append(each_target_tracked_success_rate_single)
            returns.append(rewards)
            if suc:
                success_epi += 1
                print(f"success_epi = {success_epi}")
            print('Returns is', rewards)

        plt.figure()
        plt.plot(range(len(epi_success_rate)), epi_success_rate, label='both_obs')
        for i, target in enumerate(self.env.world.targets_u):
            # 记录每个target的单个epi追踪成功率
            name = 'target' + str(i)
            plt.plot(range(len(each_target_tracked_success_rate)), np.array(each_target_tracked_success_rate)[:, i], label=name)


        plt.legend()
        plt.xlabel('epi')
        plt.ylabel('epi_success_rate')
        plt.title('success_rate in Exec')
        plt.savefig(self.save_path + '/TestEpiSuccessRate' + str(self.number) + '.png', format='png')
        plt.close()


        print("The success rate = ", float(success_epi / self.args.evaluate_episodes)) # 计算成功率
        print("The ave_epi success rate = ", np.mean(epi_success_rate)) # 计算成功率
        print("The ave_epi target tracking success rate = ", np.mean(each_target_tracked_success_rate, axis=0)) # 计算成功率

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