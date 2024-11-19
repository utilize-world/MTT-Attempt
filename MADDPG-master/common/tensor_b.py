# using tensorboard to visualize the training progress of actor network and critic network

import os
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

# log_dir = 'runs/EfficientNet_B3_experiment2'
#
# # 检查目录是否存在
# if os.path.exists(log_dir):
#     # 如果目录存在，获取目录下的所有文件和子目录列表
#     files = os.listdir(log_dir)
#
#     # 遍历目录下的文件和子目录
#     for file in files:
#         # 拼接文件的完整路径
#         file_path = os.path.join(log_dir, file)
#
#         # 判断是否为文件
#         if os.path.isfile(file_path):
#             # 如果是文件，删除该文件
#             os.remove(file_path)
#         elif os.path.isdir(file_path):
#             # 如果是目录，递归地删除目录及其下的所有文件和子目录
#             for root, dirs, files in os.walk(file_path, topdown=False):
#                 for name in files:
#                     os.remove(os.path.join(root, name))
#                 for name in dirs:
#                     os.rmdir(os.path.join(root, name))
#             os.rmdir(file_path)
#
# # 创建新的SummaryWriter
# writer = SummaryWriter(log_dir)

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter(comment='test_tensorboard')
#
# for x in range(100):
#     writer.add_scalar('y=2x', x * 2, x)
#     writer.add_scalar('y=pow(2, x)', 2 ** x, x)
#
#     writer.add_scalars('data/scalar_group', {"xsinx": x * np.sin(x),
#                                              "xcosx": x * np.cos(x),
#                                              "arctanx": np.arctan(x)}, x)
# writer.close()


class TensorboardWriter:
    def __init__(self, log_dir, args, writer=None, time_step=None):
        self.args = args
        self.writer = writer
        self.log_dir = log_dir
        self.time_step = time_step
        self.write_enabled = True  # 添加标志位来控制写入操作
        self.hist_enable = False
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
    # 创建writer
    def create_writer(self):
        self.writer = SummaryWriter(log_dir=self.log_dir)
        return self.writer

    def get_timestep(self, time_step):
        self.time_step = time_step

    # 记录网络数据
    def tensorboard_model_collect(self, wrapper, alg):
        if not self.write_enabled:
            return
        # 生成虚拟输入向量
        # 因为目的是为了看网络结构，所以输入的维度只要保证最后一维能连上就行
        if alg == "MADDPG":
            ac_input_dim = wrapper.actor.get_input_dim()
            cr_input_dim_o, cr_input_dim_a = wrapper.critic.get_input_dim()
            dummy_input_Actor = torch.randn(1, ac_input_dim).to(self.device)
            dummy_input_Critic_state = torch.randn(1, cr_input_dim_o).to(self.device)
            dummy_input_Critic_action = torch.randn(1, cr_input_dim_a).to(self.device)
            self.writer.add_graph(wrapper, (dummy_input_Critic_state, dummy_input_Critic_action, dummy_input_Actor),)
        if alg == "MLGA2C":
            ac_input_dim = wrapper.actor.input_dim
            dummy_input_Actor = torch.randn(1, ac_input_dim).to(self.device)
            dummy_input_Critic_state = torch.randn(2, 7, 2).to(self.device)
            dummy_input_Critic_state_o = [torch.randn(2, 7, 2).to(self.device), torch.randn(2, 7, 2).to(self.device)]
            dummy_input_critic_ac = torch.randn(2, 1, 2).to(self.device)
            dummy_input_critic_ac_o = [torch.randn(2, 1, 2).to(self.device),torch.randn(2, 1, 2).to(self.device)]
            self.writer.add_graph(wrapper, (dummy_input_Actor,
                                            dummy_input_Critic_state,
                                            dummy_input_Critic_state_o,
                                            dummy_input_critic_ac,
                                            dummy_input_critic_ac_o))
    # 记录常量数据
    def tensorboard_scalardata_collect(self, metrics, timestep, name=None):
        if not self.write_enabled:
            return
        self.writer.add_scalar(f"{name}_", metrics, timestep)

    # 记录parameter变化
    def tensorboard_histogram_collect(self, actor, critic, timestep, agent_id):
        if not self.write_enabled or not self.hist_enable:
            return
        # 记录 Actor 网络的参数
        for name, param in actor.named_parameters():
            self.writer.add_histogram(f'Actor{agent_id}/Parameters/{name}', param, timestep)

        # 记录 Critic 网络的参数
        for name, param in critic.named_parameters():
            self.writer.add_histogram(f'Critic{agent_id}/Parameters/{name}', param, timestep)

    # 关闭writer
    def close_writer(self):
        self.writer.close()

    # 启用写入操作
    def enable_write(self):
        self.write_enabled = True

    # 禁用写入操作
    def disable_write(self):
        self.write_enabled = False

    def enable_hist(self):
        self.hist_enable = True

    def disable_hist(self):
        self.hist_enable = False
# 用来记录各agent和target的位置和动作

class MTT_tensorboard:
    def __init__(self, agent, target, log_dir, args):
        self.args = args
        self.MTT_tensorboard = TensorboardWriter(log_dir, self.args)
        self.writer = self.MTT_tensorboard.create_writer()
        # 传入了多个agent和target
        self.agent = agent
        self.target = target

    def record_basic_info(self, current_time_step):
        for i, agent in enumerate(self.agent):

            self.writer.add_scalar("action information", {f"agent_{i}_action_x": self.agent.u[0],
                                                          f"agent_{i}_action_y": self.agent.u[1]},
                                   current_time_step)

    def close_writer(self):
        self.writer.close()