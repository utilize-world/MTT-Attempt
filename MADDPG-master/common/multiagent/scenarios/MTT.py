import numpy as np
from multiagent.core import World, Agent, Landmark, target_UAVs
from multiagent.scenario import BaseScenario
import math
from utils import find_index  # 这个函数用来奖励的计算，防止多个UAV追踪同一目标


class Scenario(BaseScenario):
    def make_world(self):
        """
        make_world是整个环境的基础，根据core中定义的类，创建了整个环境，并定义了基本的个数，实例化各实体，并初始化数据(reset_world)\n
        :return: world
        """
        world = World()
        world.comm_range = 0.4
        world.bound = 5  # 尝试改变world的规模
        # set any world properties first
        # world.dim_c = 2     # 通信维度在这里也能定义，之前在core中已经定义过为3*agent的个数，这里将其注释掉
        num_uav = 2  # 就是UAVs个数
        num_target = 1  # 这个是target个数

        num_landmarks = 0  # 没有阻挡物


        # add agents
        world.agents = [Agent() for i in range(num_uav)]
        world.targets_u = [target_UAVs() for i in range(num_target)]
        world.set_comm_dimension()
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = False
            agent.obs_range = 0.5
            agent.safe_range = 0.2
            # agent.adversary = True if i < num_adversaries else False    # 前面的为追踪者
            agent.size = 0.06  # if agent.adversary else 0.05    # ?这个尺寸有什么用
            # agent.accel = 3.0 if agent.adversary else 4.0     # 不需要加速度，可以为0
            # agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0  # if agent.adversary else 1.3   # 最大速度，速度应设置为恒定，待修改
        for i, target in enumerate(world.targets_u):
            target.name = 'target %d' % i
            target.size = 0.05
        # add landmarks 这里应该没有
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.name = 'landmark %d' % i
        #     landmark.collide = True
        #     landmark.movable = False
        #     landmark.size = 0.2
        #     landmark.boundary = False
        # # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        """
        这个是最关键的函数，是环境各变量初始化的步骤
        关于初始化的数据，仍有待考虑
        """
        bound_ad_value = 0.5  # 这个用来调整随机生成时边界值
        # 定义了各实体的颜色
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35])
        for i, target in enumerate(world.targets_u):
            target.color = np.array([0.85, 0.35, 0.35])
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # 初始化各实体状态
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(bound_ad_value, world.bound - bound_ad_value, world.dim_p)
            # agent.state.p_pos = np.random.uniform(5, 10, world.dim_p)
            agent.state.p_vel = np.random.uniform(-0.2, 0.2, 1)  # 注意这里的设置将其定义为常数
            #agent.state.move_angle = np.random.uniform(-180, 180, 1)  # 取-180到180
            # 这里修改了，move_angle和p——vel分别对应速度
            agent.state.move_angle = np.random.uniform(-0.2, 0.2, 1)
            agent.state.c = np.zeros(world.dim_c)
            agent.state.target_angle = 0
            agent.state.target_pos = np.zeros(world.dim_p)
        for target in world.targets_u:
            target.state.p_pos = np.random.uniform(bound_ad_value, world.bound - bound_ad_value, world.dim_p)
            target.state.p_vel = 0.2
            target.state.move_angle = np.random.uniform(-180, 180, 1)  # 唯一与agent的不同就是速度稍慢
            target.state.move_angle = 1    # 固定角度，用来测试
            target.state.p_pos = [0.5, 2.5]
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    # def benchmark_data(self, agent, world):
    #     # 这个是原来追逐战判断是否碰撞的，用来衡量score
    #     # 在这个环境中没用
    #     # returns data for benchmarking purposes
    #     if agent.adversary:
    #         collisions = 0
    #         for a in self.good_agents(world):
    #             if self.is_collision(a, agent):
    #                 collisions += 1
    #         return collisions
    #     else:
    #         return 0

    def is_collision(self, agent1, agent2):
        # 判断是否碰撞
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def get_agent(self, world):
        """
        返回[UAV1, UAV2, UAV3...]\n
        :param world:
        :return: [agent for agent in world.agents]
        """
        return [agent for agent in world.agents]

    # return all adversarial agents
    def get_target(self, world):
        """
        返回[target1, target2, target3...]\n
        :param world:
        :return: [target for target in world.targets_u]
        """
        return [target for target in world.targets_u]

    # 以下两个函数都用不到(是原来的Tag环境)
    # def reward(self, agent, world):
    #     # Agents are rewarded based on minimum agent distance to each landmark
    #     main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
    #     return main_reward

    # def agent_reward(self, agent, world):
    #     # Agents are negatively rewarded if caught by adversaries
    #     rew = 0
    #     shape = False
    #     adversaries = self.adversaries(world)
    #     if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
    #         for adv in adversaries:
    #             rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
    #     if agent.collide:
    #         for a in adversaries:
    #             if self.is_collision(a, agent):
    #                 rew -= 10
    # 定义agent的奖励
    def agent_reward(self, agent, world, agent_index):
        """
        用来返回agent对应的奖励\n
        这里暂且设置为，与目标距离，安全距离和边界距离 有关\n
        并且设置边界\n
        :param agent_index: agent的id
        :param agent:当前agent
        :param world:所在环境
        :return:
        """
        dis_reward = 0
        safe_reward = 0
        bound_reward = 0
        judge_rew = 3  # 奖励，用来控制边界
        #   边界奖励权重，越高则边界惩罚越大
        bound_reward_weight = 0.1
        # 如果观察到对象，返回距离奖励，否则为0
        # if agent.obs_flag:
        #     dis_map = world.distance_cal_target()
        #     min_dis = np.min(dis_map[agent_index])  # 就是对应id的那一行
        #     dis_reward = (agent.obs_range - min_dis) / agent.obs_range + 1  # 返回与距离有关的奖励
        # --------------------distance reward
        dis_map = world.distance_cal_target()
        # 这个方法是后来写的,用来更新所有的观察状态，被观察到的target将不会被其他agent观察到，就是只会被最近的agent观察到来计算奖励
        # world.update_observed_state(dis_map, agent.obs_range)
        # 所以下面有些步骤可以优化
        dis_map_agent = dis_map[agent_index]  # 就是对应id的那一行
        index_up_order = find_index(dis_map_agent)  # 返回距离由小到大的顺序index
        min_dis = np.min(dis_map_agent)
        # 如果下面几行注释掉，代表每个agent只会计算与自己距离最近的
        # 对于没有观察到target的agent，要考虑其他的未被观察的target，而不是最近的已经被观察到的
        # if world.train:
        # if agent.obs_flag is False:
        #     for value in index_up_order:
        #         if world.targets_u[int(value)].be_observed:
        #             # 如果离得最近的目标已经被观察到了，则选择次近的目标
        #             continue
        #         else:
        #             min_dis = dis_map_agent[value]
        #             break

        # 如果是在训练则返回对应的奖励，全知条件下;反之执行过程中，只有在观察到的条件下才能获得奖励
        min_dis = float(min_dis)

        dis_reward = (agent.obs_range - min_dis) / max(agent.obs_range, min_dis)  # 返回与距离有关的奖励
        dis_reward = min(math.exp(-(min_dis-agent.obs_range)), 1)
        if dis_reward < -1:
            print("abnormal reward")

        # dis_reward = -min(0.2*min_dis, 1)
        # dis_reward = float((agent.obs_range - min_dis)) / (min_dis + agent.obs_range * 0.25)  # 修改奖励
        # dis_reward = math.exp(agent.obs_range - min_dis)

        if agent.obs_flag:
            print("detected target")
            dis_reward += 5


        # distance reward-------------------

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        # 这是个静态函数
        def bound_func(x, y):
            dio = math.sqrt(
                abs(x - world.bound / 2) ** 2 + abs(y - world.bound / 2) ** 2)  # 以400,400为中心，不知道是不是这个大小，边界设置为0,800
            return dio

        if bound_func(agent.state.p_pos[0], agent.state.p_pos[1]) > (world.bound / 2):
            rew = - 0.5 * math.exp((bound_func(agent.state.p_pos[0], agent.state.p_pos[1]) -
                                    (world.bound / 2)) / bound_func(agent.state.p_pos[0],
                                                                    agent.state.p_pos[1]))  # 计算边界奖励

            # 选择exp会导致出界的惩罚非常大，所以去掉指数
            # rew = 1 - (bound_func(agent.state.p_pos[0], agent.state.p_pos[1]) -
            #            (world.bound / 2))  # 计算边界奖励
            # rew = -1 + math.exp(-(bound_func(agent.state.p_pos[0], agent.state.p_pos[1]) -
            #                       (world.bound / 2)))
            bound_reward = rew
        # 重新规定边界奖励，因为之前的是在圆形区域来考虑的
        x_bound = abs(agent.state.p_pos[0] - world.bound / 2)
        y_bound = abs(agent.state.p_pos[1] - world.bound / 2)
        if x_bound < world.bound / 2 and y_bound < world.bound / 2:
            bound_reward = 0
        else:
            bound_reward = bound_reward_weight * (- max(0, x_bound - world.bound / 2) + -max(0, y_bound - world.bound / 2))
        # 计算安全距离的奖励，每个agent都会对应三个
        distance_uavs_map = world.distance_cal_agent()  # 这样做每次都要调用一次全局的表，实在是有点浪费，但是懒得改了
        for distance in distance_uavs_map[agent_index]:
            if 0 < distance <= agent.safe_range:
                safe_reward += np.float((distance - agent.safe_range)) / np.float(agent.safe_range) - 0.3
        #   三种奖励加起来就是每个agent的奖励
        reward = dis_reward + safe_reward + bound_reward

        # 测试单步最低的奖励，在出界后到底是什么情况。

        # low_reward = reward
        # if low_reward < 0:
        #     print('negative reward')
        if safe_reward < 0:
            print("not safe detected, agent:", agent_index, 'value:', safe_reward)
        if bound_reward < 0:
            print("out of boundary, agent:", agent_index, 'value', bound_reward)

        return dis_reward + bound_reward

    # 以下是对手的奖励，这里也没用
    # def adversary_reward(self, agent, world):
    #     # Adversaries are rewarded for collisions with agents
    #     rew = 0
    #     shape = False
    #     agents = self.good_agents(world)
    #     adversaries = self.adversaries(world)
    #     if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
    #         for adv in adversaries:
    #             rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
    #     if agent.collide:
    #         for ag in agents:
    #             for adv in adversaries:
    #                 if self.is_collision(ag, adv):
    #                     rew += 10
    #     return rew

    #   以下是原来任务中定义观察的，我重写一个函数
    # def observation(self, agent, world):
    #     # get positions of all entities in this agent's reference frame
    #     entity_pos = []
    #     for entity in world.landmarks:
    #         if not entity.boundary:
    #             entity_pos.append(entity.state.p_pos - agent.state.p_pos)
    #     # communication of all other agents
    #     comm = []
    #     other_pos = []
    #     other_vel = []
    #     for other in world.agents:
    #         if other is agent: continue
    #         comm.append(other.state.c)
    #         other_pos.append(other.state.p_pos - agent.state.p_pos)
    #         if not other.adversary:
    #             other_vel.append(other.state.p_vel)
    #     return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    # 定义单个agent的观察空间,静态方法
    def observation(self, agent, world):
        """
        返回拼接后的观察格式\n
        由于agent的观察在step后都已经得到了，所以这里不用再继续操作，具体写在了core里\n
        px,py,angle,tg_px,tg_py,tg_angle, comm
        :param world:
        :param agent:
        :return: 拼接后的观察空间 1*21
        """
        comm = agent.state.c  # 每个comm的维度为10，位置二维，角度，动作二维，以及其他的agent的维度？
        target_p = agent.state.target_pos  # 2
        agent_p = agent.state.p_pos  # 2
        vel = agent.state.p_vel # 1
        obs = np.hstack((agent_p, agent.state.move_angle, target_p, agent.state.target_angle))
        obs_cat_com = np.hstack((agent_p, vel, agent.state.move_angle, target_p, agent.state.target_angle, comm))
        # 18
        # 这里先不考虑通信
        # return np.hstack((agent_p, agent.state.move_angle, target_p, agent.state.target_angle, comm))
        return obs_cat_com

    # def observation(self, agent, world):
    #     """
    #     这里实际上是要回传所有的agent的拼接数据
    #     :param agent:
    #     :param world:
    #     :return:
    #     """
    def done_judge(self, world):
        agents = self.get_agent(world)
        flag = 1
        for agent in agents:
            if agent.obs_flag:
                continue
            else:
                flag = 0
                break
        if flag:
            print("all obs")
        return flag
