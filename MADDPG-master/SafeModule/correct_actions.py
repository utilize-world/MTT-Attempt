import torch
import numpy as np
from qpsolvers import solve_qp
import ConstraintNetwork as Cons
import scipy.linalg
import scipy as sp


class Correct:
    def __init__(self, N_agents, constraint_dir, state_dim, act_dim, constraint_dim, col_margin=0.33):

        self.N_agents = N_agents
        self.total_action_dim = act_dim * self.N_agents
        self.total_state_dim = state_dim * self.N_agents
        self.solver_interventions = 0
        self.solver_infeasible = 0
        self.constraint_dim = constraint_dim
        self.total_constraint_dim = constraint_dim * self.N_agents
        self.constraint_nets = self.total_constraint_dim * [None]
        self.constraint_networks_dir = constraint_dir
        self.col_margin = col_margin


    def load_constraint_nets(self):
        for i in range(self.total_constraint_dim):
            self.constraint_nets[i] = Cons.ConstraintNetwork(self.total_state_dim, self.total_action_dim).double()
            self.constraint_nets[i].load_state_dict(torch.load(self.constraint_networks_dir
                                                               + "constraint_net_" + str(i) + ".pkl"))

    @torch.no_grad()
    def correct_actions_hard(self, state, actions, constraint):
        self.load_constraint_nets()
        actions = np.concatenate(actions)  # ndarray: 6
        state = torch.tensor(np.concatenate(state))  # tensor 30,

        # (1) Problem Variables
        # Problem specific constants
        I = np.eye(self.total_action_dim)  # 6,6
        ones = np.ones(self.total_action_dim)  # (6,)
        C = np.concatenate(constraint)  # cat 3 * (2,) --> C: (6,

        # Formulate the constraints using neural networks
        G = np.zeros([self.total_action_dim, self.total_action_dim])  # 6,6


        for i, net in enumerate(self.constraint_nets):
            G[i, :] = net(state).numpy()
        # G : tuple 6,6
        # (2) Problem Variables in QP form
        # Cost Function
        q = -actions  # 6,
        P = np.eye(self.total_action_dim)  # 6,6

        # Constraints
        A = np.concatenate([-G, I, -I])  # 18, 6
        ub = np.concatenate([C - self.col_margin, ones, ones])  # 18,
        lb = None
        # 这里的A： -G 和 C - self.col_margin就定义了不等式约束，-G是kx<=b 的 k, 下面的是b，而后面的代表了变量的上下界
        # Solve Optimization Problem
        try:
            x = solve_qp(P.astype(np.float64), q.astype(np.float64), A.astype(np.float64),
                         ub.astype(np.float64), None, None, None, None)
            if x is None:
                raise ValueError("none type detected")
        except:
            self.solver_infeasible += 1
            return actions

        # Count Solver interventions
        if np.linalg.norm(actions - x) > 1e-3:
            self.solver_interventions += 1

        return x

    @torch.no_grad()
    def correct_actions_soften(self, state, actions, constraint):
        self.load_constraint_nets()
        actions = np.concatenate(actions)
        state = torch.tensor(np.concatenate(state))
        # (1) Create solver as a globar variable
        l1_penalty = 1000

        # (2) Problem Variables
        # Problem specific constants
        I = np.eye(self.total_action_dim)
        Z = np.zeros([self.total_action_dim, self.total_action_dim])
        ones = np.ones(self.total_action_dim)
        zeros = np.zeros(self.total_action_dim)
        C = np.concatenate(constraint) - self.col_margin

        # Formulate the constraints using neural networks
        G = np.zeros([self.total_action_dim, self.total_action_dim])


        for i, net in enumerate(self.constraint_nets):
            G[i, :] = net(state).numpy()

        # (2) Problem Variables in QP form
        # Cost Function
        P = sp.linalg.block_diag(I, Z + I * 0.000001, Z + I * 0.000001)  # I为6*6的对角单位阵， 对角块,P为18*18
        q = np.concatenate([-actions, ones, zeros])  # 18

        # Constraints
        A = np.vstack((np.concatenate([-G, Z, -I], axis=1),  # 24,18   6*6 第二维拼接就是 6 *18， 然后垂直堆叠就是24* 18
                       np.concatenate([Z, Z, -I], axis=1),
                       np.concatenate([Z, -I, l1_penalty * I], axis=1),
                       np.concatenate([Z, -I, -l1_penalty * I], axis=1)))

        ub = np.concatenate((C, zeros, zeros, zeros))  # 24
        lb = None

        # Solve Optimization Problem
        try:
            x = solve_qp(P.astype(np.float64), q.astype(np.float64), A.astype(np.float64),
                         ub.astype(np.float64), None, None, None, None)
            x = x[0:(self.total_action_dim)]
        except:
            self.solver_infeasible += 1
            return actions

        # Count Solver interventions
        norm_diff = np.linalg.norm(actions - x)
        if norm_diff > 1e-3:
            self.solver_interventions += 1

        # calculating an intervetion metric
        intervention_metric = np.split(np.abs(actions - x), self.N_agents)
        intervention_metric = [np.sum(i) for i in intervention_metric]
        return x, intervention_metric

    @torch.no_grad()
    def correct_actions_CBF(self, state, actions, deltaT):
        # self.load_constraint_nets()
        # 传进来的应该是[[posx, posy, velx, vely, ...], [...] ]
        pos = state[:, :2]
        vel = state[:, 2:4]
        est_pos = pos * vel * deltaT


        actions = np.concatenate(actions)  # ndarray: 6
        state = torch.tensor(np.concatenate(state))  # tensor 30,

        # (1) Problem Variables
        # Problem specific constants
        I = np.eye(self.total_action_dim)  # 6,6
        ones = np.ones(self.total_action_dim)  # (6,)
        # C = np.concatenate(constraint)  # cat 3 * (2,) --> C: (6,

        # Formulate the constraints using neural networks
        G = np.zeros([self.total_action_dim, self.total_action_dim])  # 6,6

        for i, net in enumerate(self.constraint_nets):
            G[i, :] = net(state).numpy()
        # G : tuple 6,6
        # (2) Problem Variables in QP form
        # Cost Function
        q = -actions  # 6,
        P = np.eye(self.total_action_dim)  # 6,6

        # Constraints
        A = np.concatenate([-G, I, -I])  # 18, 6
        ub = np.concatenate([C - self.col_margin, ones, ones])  # 18,
        lb = None
        # 这里的A： -G 和 C - self.col_margin就定义了不等式约束，-G是kx<=b 的 k, 下面的是b，而后面的代表了变量的上下界
        # Solve Optimization Problem
        try:
            x = solve_qp(P.astype(np.float64), q.astype(np.float64), A.astype(np.float64),
                         ub.astype(np.float64), None, None, None, None)
            if x is None:
                raise ValueError("none type detected")
        except:
            self.solver_infeasible += 1
            return actions

        # Count Solver interventions
        if np.linalg.norm(actions - x) > 1e-3:
            self.solver_interventions += 1

        return x

    def reset_metrics(self):
        self.solver_interventions = 0
        self.solver_infeasible = 0

    def get_interventions(self):
        return self.solver_interventions

    def get_infeasible(self):
        return self.solver_infeasible
