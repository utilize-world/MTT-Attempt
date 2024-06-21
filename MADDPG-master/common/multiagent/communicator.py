# add a new type of agent named communicator, which aims to learn to connect all agent
from core import Agent


class communicator(Agent):
    # 对于通信中继来说，观察为通信范围的每个agent，动作还是移动，只不过速度
    def __init__(self):
        super(communicator, self).__init__()
        self.name = "communicator"
        self.comm_range = 1
        self.vx = 0
        self.vy = 0

    def set_comm_range(self, value):
        self.comm_range = value

    def update(self, delta_vx, delta_vy, dt):
        self.vx += delta_vx * dt
        self.vy += delta_vy * dt





    
