import os
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optimizer
import numpy as np
import pandas as pd

torch.manual_seed(2024)
np.random.seed(2024)

class ConstraintNetwork(nn.Module):
    '''
    Neural Network used to learn the constraint sensitivity.
    (Tailored to this specific set of examples)
    '''

    def __init__(self, state_dim, act_dim, hidden_size = 10, lr = 5e-4):
        '''
        Constructor
        Arguments:
            - state_dim   : the state dimension of the RL agent
            - action_dim  : the action dimension of the RL agent
            - hidden_size : hidden layer size
        '''
        super(ConstraintNetwork, self).__init__()

        # Network Architecture
        self.layer_1 = nn.Linear(state_dim, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, act_dim)
        self.layers  = [self.layer_1, self.layer_2]

        # Optimizer
        self.optimizer = optimizer.Adam(self.parameters(), lr)

        # Initialization
        self.init_weights()

    def init_weights(self):
        '''
        Weights initialization method
        '''
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
       '''
       Forward propagation method
       '''
       x = nn.ReLU()(self.layer_1(x))
       x = self.layer_2(x)
       return x

    def train(self, state, action, id, constraints_diff, epochs=100, batch_size = 256, split_ratio = 0.10):

        # Training - Validation split
        shuffle_idx = np.arange(state.shape[0])
        np.random.shuffle(shuffle_idx)

        split_idx =  int(state.shape[0]*split_ratio)
        train_idx = shuffle_idx[split_idx:]
        val_idx   = shuffle_idx[0:split_idx:]

        # Training data
        train_state       = state[train_idx,:]
        train_action      = action[train_idx,:]
        train_constraints_diff = constraints_diff[train_idx]

        # Validation data
        val_state            = state[val_idx,:]
        val_action           = action[val_idx,:]
        val_constraints_diff = constraints_diff[val_idx]

        loss_training = []
        loss_validation = []

        # Training Loop
        for epoch in range(epochs):
            for batch in range(train_state.shape[0]//batch_size):
                # Randomly select a batch
                batch_idx = np.random.choice(np.arange(train_state.shape[0]), size = batch_size)

                state_batch            = torch.Tensor(train_state[batch_idx,:])
                action_batch           = torch.Tensor(train_action[batch_idx,:])
                constraints_diff_batch = torch.Tensor(train_constraints_diff[batch_idx])

                out  = self.forward(state_batch)
                assert out.requires_grad == True
                assert out.shape[0] == batch_size

                # Loss Function
                dot_prod = torch.sum(torch.mul(out, action_batch), axis = 1)

                loss = nn.MSELoss()(constraints_diff_batch, dot_prod)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_tv = loss.detach().numpy().copy()
                loss_training.append(loss_tv)

            # Evaluate NNs performance on the validation set
            val_state            = torch.Tensor(val_state)
            val_action           = torch.Tensor(val_action)
            val_constraints_diff = torch.Tensor(val_constraints_diff)

            with torch.no_grad():
                out  = self.forward(val_state)
                dot_prod = torch.sum(torch.mul(out, val_action), axis = 1)
                loss = nn.MSELoss()(val_constraints_diff,dot_prod)
                loss_value = loss.detach().numpy().copy()
                loss_validation.append(loss_value)

            print(f"Epoch: {epoch+1}/{epochs}, val_loss {loss}")
        return loss_training, loss_validation
