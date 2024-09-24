import numpy as np
import math

import numpy.random
from utils import begin_debug
from utils import find_index
from utils import randomWalk, check_target_out_2D, limit_vel


# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position，应该包括x，y
        self.p_pos = None
        # physical velocity，应该是一个常数
        self.p_vel = None
        # 移动的角度， 常数
        self.move_angle = None


# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    """
    agent自己的状态，包括自身的位置，速度，角度，通信信息，(if have)目标位置，目标速度，目标角度(用来计算VX,VY)\n
    self.p_pos = None\n
    self.p_vel = None\n
    self.move_angle = None\n
    self.c = None  # 通信的信息， 包括xj，xj，vxj，vyj，a（t-1）j\n
    self.target_pos = None  # 包括了目标的位置信息\n
    self.target_vel = None  # 包括了目标的速度\n
    self.target_angle = None  # 目标的角度\n
    """

    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None  # 通信的信息， 包括xj，xj，vxj，vyj，a（t-1）j
        self.target_pos = None  # 包括了目标的位置信息
        self.target_vel = None  # 包括了目标的速度
        self.target_angle = None  # 目标的角度


# action of the agent
class Action(object):
    def __init__(self):
        # physical action，这里代表角速度a
        self.u = None
        # communication action 是否进行通信，即在最大通信范围内时可以通信
        self.c = None
        #


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 10
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass  # 当前任务中其实不需要考虑
        self.initial_mass = 1.0

    @property  # 只读修饰器，让这个函数不会有输入，可以直接调用mass而不是mass()
    def mass(self):
        return self.initial_mass


# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()


# properties of agent entities
class Agent(Entity):
    """
    定义Agent，即UAVs的各种属性 \n
    self.movable = True  # 是否能移动 \n
    self.silent = False  # 是否能通信 \n
    self.blind = False  # 是否能观察\n
    self.u_noise = None  # 物理噪声\n
    self.c_noise = None  # 通信噪声\n
    self.u_range = 1.0  # 动作最大值\n
    self.state = AgentState()  # state用agent的state\n
    self.action = Action()  # action用agent的action\n
    self.action_callback = None\n
    # 观察flag\n
    self.obs_flag = False\n
    # 观察范围\n
    self.obs_range = 60\n
    # 安全距离\n
    self.safe_range = 30 \n
    而AgentState()\n
    agent自己的状态，包括自身的位置，速度，角度，通信信息，(if have)目标位置，目标速度，目标角度(用来计算VX,VY)\n
    self.p_pos = None\n
    self.p_vel = None\n
    self.move_angle = None\n
    self.c = None  # 通信的信息， 包括xj，xj，vxj，vyj，a（t-1）j\n
    self.target_pos = None  # 包括了目标的位置信息\n
    self.target_vel = None  # 包括了目标的速度\n
    self.target_angle = None  # 目标的角度\n
    """

    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True  # 是否能移动
        # cannot send communication signals
        self.silent = False  # 是否能通信
        # cannot observe the world
        self.blind = False  # 是否能观察
        # physical motor noise amount
        self.u_noise = None  # 物理噪声
        # communication noise amount
        self.c_noise = None  # 通信噪声
        # control range
        self.u_range = 1.0  # 动作最大值
        # state
        self.state = AgentState()  # state用agent的state
        # action
        self.action = Action()  # action用agent的action
        # script behavior to execute
        self.action_callback = None

        # 观察flag
        self.obs_flag = False
        # 观察范围
        self.obs_range = 3
        # 安全距离
        self.safe_range = 1


# 用来指定那些target
class target_UAVs(Agent):
    """
    继承了Agent的属性\n
    在此基础上，将通信和动作设置为None \n
    self.action = None\n
    self.silent = True  # 不能通信，不能决策\n
    """

    def __init__(self):
        super(target_UAVs, self).__init__()
        self.name = "target"
        self.action = None
        self.silent = True  # 不能通信，不能决策
        self.be_observed = False  # 定义了自己的状态，是否被观察到
        self.out = False  # 定义是否出界


# multi-agent world
class World(object):  # 最关键的
    """
    定义了基本的属性
    # list of agents and entities (can change at execution-time!)
        self.agents = [] # UAVs\n
        self.landmarks = [] # 阻挡物\n
        self.targets_u = []  # target  \n
        通信信息有多少种数据？ 包括了当前位置，当前速度和上一次动作 \n
        self.dim_c = 4 * len(self.agents)   # 通信维度\n
        self.dim_p = 2  # 位置维度\n
        self.dim_color = 3 # 颜色维度\n
        self.dt = 0.1   # 模拟时间间隔\n
        self.damping = 0.25 # 物理阻尼，这里没用到\n
        self.contact_force = 1e+2   # 这俩没什么用\n
        self.contact_margin = 1e-3

        self.comm_map = np.zeros((len(self.agents), len(self.agents)))  # 通信表，用来表示与所有agent的通信的逻辑关系\n

        self.comm_range = 100   # 通信范围\n
        self.bound = 1000 # 边界\n

        self.dim_ac = 4 # 定义通信动作（信息）的维度
    """

    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.targets_u = []  # 无人机
        # communication channel dimensionality,通信信息有多少种数据？ 包括了当前位置，当前速度和上一次动作
        self.dim_c = 12
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # 通信表，用来表示与所有agent的通信的逻辑关系
        self.comm_map = np.zeros((len(self.agents), len(self.agents)))
        # 通信范围
        self.comm_range = 0.5
        # 边
        self.bound = 2
        self.rebound = 0.04
        self.dim_ac = 6
        self.Na = 20
        # 定义是否在训练，这与状态有关
        self.train = True
        # Spacial Entropy distance
        self.SE_dis = self.bound / 4
        self.update_time = 0  # 用于target更新计时
        self.vel_bound = 0.08   # limit velo
        # 以下用于target出界检测
        self.direction_changed_flag = [False]*len(self.targets_u) # 用于target出界检测，是否变动过角度？
        self.t_out_bound = False    # 用于记录target是否出界

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.targets_u + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # 设置当前的通信维度
    def set_comm_dimension(self):
        self.dim_c = len(self.agents) * self.dim_ac

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    ##############
    # 自己写的方法，用来环境更新状态，由于不再有力的参与，所以应该相对简单
    def env_step(self):
        """
        包括环境中目标位置的更新\n
        UAVs的位置，角度的更新\n
        对UAVs状态的更新，包括上面的更新以及通信信息，观察信息的更新，这些都没有拼接，而是作为本身的属性保留\n
        通信表的更新，距离表的更新，目标是否在范围的检测(因为与状态相关)\n
        :return: None
        """
        low_rebound = self.rebound
        high_rebound = self.bound - low_rebound  # 这两个参数用来控制target的回弹

        for i, target in enumerate(self.targets_u):
            # 当前target是否出界，如果出界，并且已经更改过了角度，则不再更改角度

            if not self.direction_changed_flag[i]:
                # 位置更新,目标的位置
                if (target.state.p_pos[0] < low_rebound and target.state.move_angle > 90) or \
                        (target.state.p_pos[0] > high_rebound and 0 < target.state.move_angle < 90):
                    target.state.move_angle = (180 - target.state.move_angle)  # 反弹
                    self.direction_changed_flag[i] = True
                if (target.state.p_pos[0] < low_rebound and target.state.move_angle < -90) or \
                        (target.state.p_pos[0] > high_rebound and 0 > target.state.move_angle > -90):
                    target.state.move_angle = -(180 + target.state.move_angle)
                    self.direction_changed_flag[i] = True
                if target.state.p_pos[1] < low_rebound and target.state.move_angle < 0:
                    target.state.move_angle *= -1
                    self.direction_changed_flag[i] = True
                if target.state.p_pos[1] > high_rebound and target.state.move_angle > 0:
                    target.state.move_angle *= -1
                    self.direction_changed_flag[i] = True
            # 固定方向运动
            # 固定方向为指定角度：一直保持

            # test——code
            target.state.p_pos[0] += target.state.p_vel * math.cos(target.state.move_angle *
                                                                   (math.pi / 180)) * self.dt
            target.state.p_pos[1] += target.state.p_vel * math.sin(target.state.move_angle *
                                                                   (math.pi / 180)) * self.dt
            self.t_out_bound = check_target_out_2D(target.state.p_pos, [low_rebound, high_rebound],
                                                   [low_rebound, high_rebound])
            # 检查是否在界内？如果在界内，就可以重新开始出界检查，否则继续认为出界，角度和随机游走都不再发生
            if not self.t_out_bound:
                self.direction_changed_flag[i] = False
            # 随机游动，这里是固定时间间隔就会变化一次9
            # TODO:target moving policy
            # 如果角度还没改变，才能使用随机游走，否则表明已经出界，就别游走了
            if not (self.direction_changed_flag[i]) and (self.update_time % 50 == 0) and (self.update_time > 0):
                target.state.p_vel, target.state.move_angle = randomWalk(target.state.p_vel,
                                                                         target.state.move_angle,
                                                                         0.01,
                                                                         90
                                                                         )
                self.update_time = 0
            else:
                self.update_time += 1

            # if not target.out:    # 如果没出界
            #     target.state.move_angle += numpy.random.uniform(-180, 180) / 8  # 每步最大变化±π/8
            # target.state.p_pos[0] += target.state.p_vel * math.cos(target.state.move_angle *
            #                                                        (math.pi / 180)) * self.dt
            # target.state.p_pos[1] += target.state.p_vel * math.sin(target.state.move_angle *
            #                                                        (math.pi / 180)) * self.dt
            if target.state.p_pos[0] > self.bound or target.state.p_pos[1] > self.bound:
                print("target go bound out")
                target.out = True
            elif target.state.p_pos[0] < 0 or target.state.p_pos[1] < 0:
                print("target go zero out")
                target.out = True
            else:
                target.out = False
            # 对agent也就是无人机进行更新，位置，角度和
        for i, agent in enumerate(self.agents):
            # agent.state.move_angle += np.float((2 * agent.action.u[0] - self.Na - 1)) / (self.Na - 1) * 0.02 * self.dt
            # agent.state.p_vel += np.float((2 * agent.action.u[1] - self.Na - 1)) / (
            #             self.Na - 1) * 0.02 * self.dt  # 最大加速度为±5
            agent.state.move_angle += agent.action.u[0] * self.dt     # directly apply the u to velocity control
            agent.state.p_vel += agent.action.u[1] * self.dt
            #  TODO: xiansu
            agent.state.move_angle, agent.state.p_vel = limit_vel(agent.state.move_angle, self.vel_bound), \
                                                        limit_vel(agent.state.p_vel, self.vel_bound)
            #

            # agent.state.p_pos[0] += agent.state.p_vel * math.cos(agent.state.move_angle *
            #                                                      (math.pi / 180)) * self.dt
            # agent.state.p_pos[1] += agent.state.p_vel * math.sin(agent.state.move_angle *
            #                                                      (math.pi / 180)) * self.dt
            # 更改运动模式，action[0]和action[1]作为x，y轴加速度，而move——angle和p——vel对应x，y轴速度
            agent.state.p_pos[0] += agent.state.move_angle * self.dt
            agent.state.p_pos[1] += agent.state.p_vel * self.dt
            j = 0
            # set communication state (directly for now)
            # 如果在通信范围内，则可以获得其他所有agent的通信动作(也就是通信信息)
            # 根据通信表       UAV\UAV      1   2   3
            #                    1        F   ?   ?
            #                    2        ?   F   ?
            #                    3        ?   ?   F
            #
            agent.state.c = []
            if agent.silent:
                # agent.state.c = np.zeros(self.dim_c)  # 有多少个agent就会有多少个通信内容
                pass

            else:
                # 判断其余的agent是否在通信范围内
                # 更新通信表
                self.comm_map = self.update_comm_map(self.distance_cal_agent())
                # if np.any(self.comm_map == 1):
                #     print("comm detected")
                #### test_code
                # begin_debug(True in self.comm_map)

                ####
                for flag in self.comm_map[i]:
                    # 检查每一行，自己肯定是0，j代表这一行对应的其他的agent的下标
                    if flag:
                        noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
                        agent.state.c = np.hstack(
                            (agent.state.c, (self.agents[j].action.c + noise)))  # 如果在, 就把所有通信内容加起来拼接    # 通信应该要
                    else:
                        agent.state.c = np.hstack((agent.state.c, (np.zeros(self.dim_ac))))  # 否则就是0
                    j += 1
            #  判断是否有target在范围中，并选择最近的作为状态信息，这个最近的前提，是未被观察到的agent的最近
            #  这里存在争议，
            distance_a_t_map = self.distance_cal_target()
            min_dist, min_index = np.min(distance_a_t_map[i]), np.argmin(distance_a_t_map[i])
            # if agent.obs_flag is False:
            #     dist_up_index = find_index(distance_a_t_map[i])
            #     for value in dist_up_index:
            #         if self.targets_u[int(value)].be_observed:
            #             continue
            #         else:
            #             min_index = int(value)
            #             break
            # if self.train is False:
            #     # 如果在执行阶段，只有范围内的数据才能得到
            #     if min_dist < agent.obs_range:
            #         agent.state.target_pos = self.targets_u[min_index].state.p_pos
            #         agent.state.target_vel = self.targets_u[min_index].state.p_vel
            #         agent.state.target_angle = self.targets_u[min_index].state.move_angle
            #     else:
            #         agent.state.target_pos = np.zeros(self.dim_p)
            #         agent.state.target_vel = 0
            #         agent.state.target_angle = 0
            # else:
            #     # 如果在训练阶段，所有的state都是可知的
            #     agent.state.target_pos = self.targets_u[min_index].state.p_pos
            #     agent.state.target_vel = self.targets_u[min_index].state.p_vel
            #     agent.state.target_angle = self.targets_u[min_index].state.move_angle

            # 假设所有情况下都知道目标的位置
            agent.state.target_pos = self.targets_u[min_index].state.p_pos
            agent.state.target_vel = self.targets_u[min_index].state.p_vel
            agent.state.target_angle = self.targets_u[min_index].state.move_angle

    # 写出计算距离公式, 更新距离表(agent之间的)
    # 根据距离表       UAV\UAV      1   2   3
    #                    1        0   ?   ?
    #                    2        ?   0   ?
    #                    3        ?   ?   0
    def distance_cal_agent(self):
        distance_map = np.zeros((len(self.agents), len(self.agents)))
        for i in range(len(self.agents)):
            for j in range(len(self.agents)):
                if i == j:
                    distance_map[i][j] = 0
                else:
                    distance_map[i][j] = np.square(self.agents[i].state.p_pos[0] - self.agents[j].state.p_pos[0]) + \
                                         np.square(self.agents[i].state.p_pos[1] - self.agents[j].state.p_pos[1])
        distance_map = np.sqrt(np.array(distance_map))  # 求出欧式距离
        # print(list(distance_map))  # 测试
        return distance_map

    # 计算目标距离(当前目标范围内)
    # 写出计算距离公式, 更新距离表(agent之间的)
    # 根据距离表       UAV\tar      1   2   3
    #                    1        ?   ?   ?
    #                    2        ?   ?   ?
    #                    3        ?   ?   ?
    def distance_cal_target(self):
        distance_a_t_map = np.zeros((len(self.agents), len(self.targets_u)))
        dm = []
        # 关于reshape的测试
        # test_a = np.array([[1, 2, 3],
        #                    [4, 5, 6]])
        # t = test_a.reshape(6, )
        # print(t)
        # print(t.reshape((2, 3)))
        # 输出
        # out: [1 2 3 4 5 6]
        # [[1 2 3]
        #  [4 5 6]]
        for agent in self.agents:
            agent.obs_flag = False
            for index, target in enumerate(self.targets_u):
                # if not agent.obs_flag:  # 如果当前没有观察到目标，则查看距离，并且修改自己的obs_flag
                distance_a_t = np.sqrt(np.square(agent.state.p_pos[0] - target.state.p_pos[0]) +
                                       np.square(agent.state.p_pos[1] - target.state.p_pos[1]))
                dm.append(distance_a_t)
                if distance_a_t <= agent.obs_range:
                    # if distance_a_t < agent.obs_range and target.be_observed == False:
                    agent.obs_flag = True
                    # target.be_observed = True
                else:
                    agent.obs_flag = False
        dm = np.array(dm)
        distance_a_t_map = dm.reshape((len(self.agents), len(self.targets_u)))  # 最终会生成距离表
        return distance_a_t_map

    # 通信表更新
    def update_comm_map(self, distance_map):
        comm_map = []
        distance_map_row = len(distance_map)
        distance_map_col = len(distance_map[0])
        dm_r = distance_map.reshape((distance_map_row * distance_map_col), )  # 把距离矩阵变成一维数组
        for i in dm_r:
            comm_map.append([True if 0 < i < self.comm_range else False])
        comm_map = np.array(comm_map).reshape((distance_map_row, distance_map_col))
        return comm_map

    #######

    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if (b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if (f_a is not None):
                    if (p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if (f_b is not None):
                    if (p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)  # damping物理阻尼
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                      np.square(
                                                                          entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

            # get collision forces for any contact between two entities

    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if (entity_a is entity_b):
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    # 定义一个目标被观察状态，以及，agent的观察的表
    def get_obs_state(self, distance_agent_target_map, obs_dis):
        # 定义目标被观察的情况表
        #   target_be_observed_map = [  0   0   0
        #                               0   0   0
        #                               0   0   0   # 第i，j代表target[i]被agent(j)观察
        #                            ]
        #   最后返回一个judgemap，映射了观察情况
        target_be_observed_map = np.zeros(((len(self.targets_u)), len(self.agents)))  # target行, agent列
        judge_map = []
        dis_map = distance_agent_target_map.copy()

        maxvalue = np.max(dis_map)
        for i in range(len(self.targets_u)):
            indexes = np.unravel_index(np.argmin(dis_map), dis_map.shape)  # agent行，target列(个数)
            test_value = dis_map[indexes]
            test_index = indexes
            if test_value < obs_dis:
                target_be_observed_map[test_index[1], test_index[0]] = 1
                # 把这一行这一列全部设置为最大值
                # 应该要掩盖一列就够了，即target可以被多个agent观察到，
                # dis_map[indexes[0]].fill(maxvalue)  # 行
                dis_map[:, indexes[1]].fill(maxvalue)  # 列，这样就可以让每个agent最多观察到一个
            else:
                break

        for i in target_be_observed_map:
            judge_map.append(i == 1)

        return judge_map

    def update_observed_state(self, distance_a_t_map, obs_dis):
        judge_map = self.get_obs_state(distance_a_t_map, obs_dis)
        for target_index, value in enumerate(judge_map):
            for agent_index, a_value in enumerate(judge_map[target_index]):
                self.targets_u[target_index].be_observed = (a_value == 1)
                self.agents[agent_index].obs_flag = (a_value == 1)
