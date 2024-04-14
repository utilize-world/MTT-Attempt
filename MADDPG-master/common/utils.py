import numpy as np
import inspect
import functools


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
    env = MultiAgentEnv(world, scenario.reset_world, scenario.agent_reward, scenario.observation, None, scenario.done_judge)
    # env = MultiAgentEnv(world)
    args.n_players = env.n  # 包含敌人的所有玩家个数
    args.n_agents = env.n - args.num_adversaries  # 需要操控的玩家个数，虽然敌人也可以控制，但是双方都学习的话需要不同的算法
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # 每一个元素代表该agent的obs维度
    action_shape = []
    for content in env.action_space:
        action_shape.append(content)
    args.action_shape = action_shape[:args.n_agents]  # 每一维代表该agent的act维度
    args.action_shape = [2 for i in range(args.n_agents)]  # 这里直接设置动作维度为1，所以前面的可能没用了
    args.high_action = world.Na
    args.low_action = 1
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
    # 如果全部都出界，则为True
    for agent in agents_array:
        is_bound.append((agent.state.p_pos.any() > max_bound) or (agent.state.p_pos.any() < low_bound))
    for i in range(len(is_bound)):
        if not is_bound[i]:
            is_bound_all = False
    return is_bound_all


import shutil
import os

def clear_folder(folder_path):
    shutil.rmtree(folder_path)
    os.mkdir(folder_path)