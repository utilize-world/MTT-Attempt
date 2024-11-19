import numpy as np
import inspect
import functools
import matplotlib.pyplot as plt
import random

from functools import lru_cache  # 缓存管理
import itertools


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def make_env(args):
    """

    :param args: arguments
    :return: env, arguments
    """
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(args.scenario_name + ".py").Scenario()

    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.agent_reward, scenario.observation, None,
                        scenario.done_judge)
    # env = MultiAgentEnv(world)
    args.n_players = env.n  # 包含敌人的所有玩家个数
    args.n_agents = env.n - args.num_adversaries  # 需要操控的玩家个数，虽然敌人也可以控制，但是双方都学习的话需要不同的算法
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # 每一个元素代表该agent的obs维度
    args.n_targets = env.n_targets
    action_shape = []
    for content in env.action_space:
        action_shape.append(content)
    args.action_shape = action_shape[:args.n_agents]  # 每一维代表该agent的act维度
    args.action_shape = [2 for i in range(args.n_agents)]  # 这里直接设置动作维度为1，所以前面的可能没用了
    args.high_action = 0.1
    args.low_action = -0.1
    # args.high_action = world.Na       # directly define the max_accelaration
    # args.low_action = 1
    return env, args


def find_index(distance_map):  # 将distance_map中的数升序排列，返回原下标
    ordered_map = sorted(distance_map)
    up_index = []
    for i in range(len(distance_map)):
        # up_index.append(distance_map.index(ordered_map[i])) narray类型没有index方法
        up_index.append(np.array(np.where(distance_map == ordered_map[i])))
    return up_index


def check_agent_bound(agents_array, max_bound, low_bound):
    is_bound = []
    is_bound_all = True
    ab = 0.1
    # 如果全部都出界，则为True
    for agent in agents_array:
        max_flag = [x > max_bound + ab for x in agent.state.p_pos]
        min_flag = [x < low_bound - ab for x in agent.state.p_pos]
        if True in max_flag or (True in min_flag):
            out = True
        else:
            out = False
        is_bound.append(out)

    if False in is_bound:
        is_bound_all = False
    return is_bound_all


def check_agent_near_bound(agents_array, max_bound, low_bound):
    is_bound = []
    is_bound_all = True
    # 如果全部都出界，则为True
    for agent in agents_array:
        max_flag = [x > max_bound for x in agent.state.p_pos]
        min_flag = [x < low_bound for x in agent.state.p_pos]
        if True in max_flag or (True in min_flag):
            out = True
        else:
            out = False
        is_bound.append(out)

    if False in is_bound:
        is_bound_all = False
    return is_bound_all


import shutil
import os


def clear_folder(folder_path):
    shutil.rmtree(folder_path)
    os.mkdir(folder_path)


# 用来设置调试的位置
def begin_debug(condition):
    import pdb
    if condition:
        pdb.set_trace()


def ezDrawAPic(saved_path, saved_name, x_label, y_label, x_data, y_data):
    plt.figure()
    plt.plot(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(saved_name)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    plt.savefig(saved_path + '/' + saved_name + '.png', format='png')
    plt.close()


def randomWalk(origin_x, origin_y, x_delta_max, y_delta_max):
    delta_X = random.uniform(-x_delta_max, x_delta_max)
    delta_Y = random.uniform(-y_delta_max, y_delta_max)
    return origin_x + delta_X, origin_y + delta_Y


def check_target_out_2D(target_pos, bound_x, bound_y):
    # target的位置是二维的，[x,y]
    # bound_x表示x轴的[xmin, xmax]
    if bound_x[0] < target_pos[0] < bound_x[1] and bound_y[0] < target_pos[1] < bound_y[1]:
        return False
        # 表示没出界，否则出界
    else:
        return True


def cal_dis(pos1, pos2):
    dis = np.sqrt(np.square(pos1[0] - pos2[0]) + np.square(pos1[1] - pos2[1]))
    return dis


def cal_relative_pos(pos1, pos2):
    return pos2 - pos1


# 检查梯度更新状态
def print_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} gradient: {param.grad.data}")
        else:
            print(f"{name} gradient: None")


# 在每个episode结束时都要归零，

# 然后是检查，如果连续judge_value个时间步，目标都被观察到，就结束，认为该episode完成，为成功的epi
def is_success(current_cumulate_tracking_timestep, judge_value=5):
    if current_cumulate_tracking_timestep >= judge_value:
        return True
    else:
        return False

# 计算agent侧的成功率，这里考虑的是整个测试(1个epi)中所有agent都能观察到任意target的时间步 / 总时间步
def is_agents_success_track(dis_map, bound_value):
    for row in dis_map:
        if all([value > bound_value for value in row]):
            # 如果条件满足，就代表当前存在agent没有观察到target
            return False
    # 循环结束都没有return,就代表所有agent都观察到至少一个target
    return True

# 判断是否发生了碰撞，碰撞率的计算还是发生碰撞的时间步 / 总的时间步
def is_collision(uav_map, bound_value):
    for row in uav_map:
        if any([0 < value <= bound_value for value in row]):
            # 如果条件满足，代表有碰撞存在
            return True
    # 循环结束都没有return,就代表没有碰撞
    return False




# 限速
def limit_vel(current_value, limited_value):
    if current_value >= limited_value:
        current_value = limited_value
    if current_value <= -limited_value:
        current_value = -limited_value
    return current_value

# 判断done中的是否全1，全1代表全部target都被追踪
def all_ones(lst):
    return all(x == 1 for x in lst)


def normNegetive(lst):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(lst - np.max(lst))  # Subtract max(x) for numerical stability
    return e_x / e_x.sum()


# 如果一个列表中都为1则返回ture否则false

from math import factorial

# 缓存的目的是为了让相同输入情况下不再重新计算
@lru_cache
def calculate_shapley_value(n, v):
    """
    计算n个参与者的Shapley值。

    参数:
    n (int): 参与者的总数。
    v (function): 特征函数，接受一个联盟的集合，返回该联盟的收益。
    dis_u_t: n*m[][] uav和target的距离数组
    dis_u_u: n*n[][] uav和uav的距离

    返回:
    list: 包含每个参与者Shapley值的列表。
    """
    # 按道理说，每次shapley计算时应当是当前通信范围内
    shapley_values = [0] * n

    for i in range(n):
        # 遍历每个可能的子集大小 r (从0到n-1)
        for r in range(n):
            for S in itertools.combinations(range(n), r):
                # 0到n-1所有可能的组合，r是多少个组合在一起
                if i not in S:
                    S_with_i = S + (i,)
                    # Shapley值的权重
                    weight = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                    # 计算边际贡献
                    marginal_contribution = v(S_with_i) - v(S)
                    shapley_values[i] += weight * marginal_contribution

    return shapley_values


# 示例特征函数
def v(S, array, dis_UAV_in_comm_index, bound):
    """
    计算联盟S的收益（返回平均值）。

    参数:
    S (tuple): 联盟的集合。
    array (np.array): 二维数组。

    返回:
    float: 联盟S的收益（平均值）。
    """
    if len(S) == 0:
        return 0
    # 对联盟S中的每一行求和，并返回总和的平均值
    else:
        value_s = 0
        for eles in S:
            for ele in array[dis_UAV_in_comm_index[eles]]:
                if ele < bound:
                    value_s -= ele
            # for dis in u_u_map[eles]:
            #     if dis < bound:
            #         value_s -= 1

    return value_s / len(S)