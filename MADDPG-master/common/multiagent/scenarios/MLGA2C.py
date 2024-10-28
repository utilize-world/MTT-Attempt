import numpy as np
from multiagent.core import World, Agent, Landmark, target_UAVs
from multiagent.scenario import BaseScenario
import math
from utils import find_index  # 这个函数用来奖励的计算，防止多个UAV追踪同一目标
from utils import cal_dis, cal_relative_pos, calculate_shapley_value, v, normNegetive
import sys
import torch.nn.functional as F

sys.path.append('../')


# from collect_Info_for_att import get_target_info


# MLGA2C中的环境，为了简化，将原本的5v5变成3v3


def calculate_relative_position_of_other_agents(agent, other, world, fortarget=False):
    relative_position_map = []
    # 假设agent j的位置为pj，agent i的位置为pi， 则相对位置为|pj-pi|
    pi = agent.state.p_pos
    if not fortarget:
        for other_agent in other:
            pj = other_agent.state.p_pos
            # 如果在通信范围，则拼接上去，否则不拼接

            if 0 < cal_dis(pi, pj) <= world.comm_range:
                relative_position_map.append(cal_relative_pos(pi, pj))
            elif cal_dis(pi, pj) > world.comm_range:
                relative_position_map.append([0, 0])
    else:
        for other_agent in other:
            pj = other_agent.state.p_pos
            if 0 < cal_dis(pi, pj) <= agent.obs_range:
                relative_position_map.append(cal_relative_pos(pi, pj))
            elif cal_dis(pi, pj) > agent.obs_range:
                relative_position_map.append([0, 0])
            # relative_position_map.append(cal_relative_pos(pi, pj))
    relative_position_map = np.array(relative_position_map).flatten()
    return relative_position_map


class Scenario(BaseScenario):
    def make_world(self):
        """
        make_world是整个环境的基础，根据core中定义的类，创建了整个环境，并定义了基本的个数，实例化各实体，并初始化数据(reset_world)\n
        :return: world
        """

        world = World()
        world.comm_range = 0.5  # 原文中是获取最近的两个agent的数据，以及最近两个target的信息
        world.comm_range = 0.5  # 全通
        world.bound = 2  # 2mx2m

        # set any world properties first
        # world.dim_c = 2     # 通信维度在这里也能定义，之前在core中已经定义过为3*agent的个数，这里将其注释掉
        num_uav = 3  # 就是UAVs个数
        num_target = 3  # 这个是target个数

        num_landmarks = 0  # 没有阻挡物

        # 以下是对agent和target属性的修改，而其基本属性在core中定义。
        world.agents = [Agent() for i in range(num_uav)]
        world.targets_u = [target_UAVs() for i in range(num_target)]
        world.set_comm_dimension()
        world.direction_changed_flag = [False] * len(world.targets_u)
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = False
            agent.obs_range = 0.3  # 观察区域为0.3m，在这个范围内，就不会获得惩罚
            agent.safe_range = 0.15  # 安全区域也为0.3m，这个范围内就不安全，会受到碰撞惩罚
            # agent.adversary = True if i < num_adversaries else False    # 前面的为追踪者
            agent.size = 0.05  # if agent.adversary else 0.05    # ?这个尺寸有什么用
            # agent.accel = 3.0 if agent.adversary else 4.0     # 不需要加速度，可以为0
            # agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0  # if agent.adversary else 1.3   # 最大速度，速度应设置为恒定，待修改
        for i, target in enumerate(world.targets_u):
            target.name = 'target %d' % i
            target.size = 0.05

        world.landmarks = [Landmark() for i in range(num_landmarks)]
        self.reset_world(world)
        return world

    def reset_world(self, world):
        """
        这个是最关键的函数，是环境各变量初始化的步骤
        关于初始化的数据，仍有待考虑
        """
        bound_ad_value = 0.1  # 这个用来调整随机生成时边界值
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
            agent.state.p_vel = np.random.uniform(-0.04, 0.04, 1)  # 注意这里的设置将其定义为常数
            # agent.state.move_angle = np.random.uniform(-180, 180, 1)  # 取-180到180
            # 这里修改了，move_angle和p——vel分别对应速度
            agent.state.move_angle = np.random.uniform(-0.04, 0.04, 1)
            agent.state.c = np.zeros(world.dim_c)
            agent.state.target_angle = 0
            agent.state.target_pos = np.zeros(world.dim_p)
        for target in world.targets_u:
            target.state.p_pos = np.random.uniform(bound_ad_value, world.bound - bound_ad_value, world.dim_p)
            target.state.p_vel = 0.04
            target.state.move_angle = np.random.uniform(-180, 180, 1)  # 唯一与agent的不同就是速度稍慢

        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

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

    def agent_reward(self, agent, world, agent_index):
        """
        用来返回agent对应的奖励\n
        这里暂且设置为，与目标距离，安全距离和时间 有关\n
        并且设置边界\n
        :param agent_index: agent的id
        :param agent:当前agent
        :param world:所在环境
        :return:
        """
        #  ------------------initialize and preprocess
        w1, w2, w3, w4, w5 = 1, 1, 1, 1, 1
        SE_reward = 0  # 空间熵奖励，这个应该随时间步逐渐消失，在这里先不定义
        dis_reward = 0  # 距离和奖励
        dis_weight = 1  # 距离和权重
        shapleyReward = 0
        bound_reward = 0
        collision_reward = 0  # 如果与其他无人机碰撞，则会获得惩罚，这一点只需要先判断是否有通信存在，然后再判断最近
        time_reward = -0.01  # 如果目标在感知范围内，则为0，否则就是一个惩罚时间的奖励
        dis_map = world.distance_cal_target()  # 得到一个所有UAV相对所有target的距离矩阵
        distance_uavs_map = world.distance_cal_agent()  # 这样做每次都要调用一次全局的表，实在是有点浪费，但是懒得改了

        dis_UAV_in_comm = []
        dis_UAV_in_comm_index = []
        for index, dis_uav in enumerate(distance_uavs_map[agent_index]):
            if dis_uav < world.comm_range:
                dis_UAV_in_comm.append(dis_uav)
                dis_UAV_in_comm_index.append(index)
        n = len(dis_UAV_in_comm)
        sha_index = dis_UAV_in_comm_index.index(agent_index)
        #     -------------------------distance sum and time---------------

        current_uav_target_dismap = dis_map[agent_index]
        min_dis = np.min(current_uav_target_dismap)
        if min_dis <= agent.obs_range:
            time_reward = 1

        #    ----global dis reward----
        for ele in dis_map:
            for dis in ele:
                if dis < agent.obs_range:
                    dis_reward += 1 + (agent.obs_range - dis) / agent.obs_range
        # shapleyReward = normNegetive(calculate_shapley_value(len(world.agents), lambda S: v(S, dis_map)))[agent_index]
        shapleyReward = calculate_shapley_value(n, lambda S: v(S, dis_map,dis_UAV_in_comm_index, agent.obs_range))[sha_index]

        #     -------------------------collision reward-----------
        # if (0 < distance <= agent.safe_range for distance in distance_uavs_map[agent_index]):
        #     collision_reward = -1
        for distance_U in distance_uavs_map:
            for disU in distance_U:
                if 0 < disU < agent.safe_range:
                    collision_reward += (disU - agent.safe_range) / agent.safe_range
        # ---bound
        if agent.state.p_pos[0] < 0 or agent.state.p_pos[0] > world.bound:
            bound_reward -= abs(-1 + agent.state.p_pos[0])
        if agent.state.p_pos[1] < 0 or agent.state.p_pos[1] > world.bound:
            bound_reward -= abs(-1 + agent.state.p_pos[1])

        if bound_reward != 0:
            print("out!b_reward=", bound_reward)
        # if agent.obs_flag:
        #     print("detected target")
        ### SE_reward----------------------------------------------------

        if n >= 2:
            ave_dis = 2 * sum(dis_UAV_in_comm) / (n * (n - 1))
            SE_reward = (ave_dis - agent.safe_range) / (world.SE_dis - agent.safe_range)
        else:
            SE_reward = 0

        # return dis_reward + collision_reward + time_reward
        return w1 * dis_reward + w2 * shapleyReward + w3 * collision_reward +\
               w4 * time_reward + 0.1 * SE_reward + w5 * bound_reward

    # 定义单个agent的观察空间,静态方法
    def observation(self, agent, world):
        """
        在这里我们仅需要知道，自身的全局位置（2dim，全局速度（2dim，所有target的相对位置（target_num * 2,
        通信范围内的agent相对位置（agent_num * 2,不在通信范围内的就补0

        因此总维度为 2 + 2 + 2n + 2m
        :param world:
        :param agent:
        :return: 拼接后的观察空间 1*21
        """
        # comm = agent.state.c  # 每个comm的维度为6,
        # target_p = agent.state.target_pos  # 2

        agent_p = agent.state.p_pos  # 2
        velx = agent.state.p_vel  # 1
        vely = agent.state.move_angle  # 1
        # temp = []
        # for target in world.targets_u:
        #     temp.append(target.state.p_pos)
        target_relative_pos = calculate_relative_position_of_other_agents(agent, world.targets_u, world, fortarget=True)
        agent_relative_pos = calculate_relative_position_of_other_agents(agent, world.agents, world)
        obs_cat = np.hstack((agent_p, velx, vely, target_relative_pos, agent_relative_pos))

        return obs_cat

    def done_judge(self, world):
        agent = self.get_agent(world)[0]
        flag = 1
        dis_map = world.distance_cal_target()
        # 所有target都被观察才是对的
        for col in list(zip(*dis_map)):
            is_target_detected = False
            for col_ele in col:
                if col_ele <= agent.obs_range:
                    is_target_detected = True
                    break
            if not is_target_detected:
                return 0

        if flag:
            print("all targets obs")
        return flag
