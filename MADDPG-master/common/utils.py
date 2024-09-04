import numpy as np
import inspect
import functools
import matplotlib.pyplot as plt
import random


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
    args.high_action = world.Na
    args.low_action = 1
    # args.high_action = world.Na       # directly define the max_accelaration
    # args.low_action = -world.Na
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
