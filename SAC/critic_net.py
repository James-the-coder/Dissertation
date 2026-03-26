import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, half_cheetah=False):
        super(Critic, self).__init__()
        self.half_cheetah = half_cheetah
        input_dim = state_dim + goal_dim + action_dim
        if self.half_cheetah:
            input_dim = state_dim + action_dim
        # defining 2 networks for clipped double Q learning
        self.net1_l1 = nn.Linear(input_dim, 256)
        self.net1_l2 = nn.Linear(256, 256)
        self.net1_l3 = nn.Linear(256, 256)
        self.net1_l4 = nn.Linear(256, 1)

        self.net2_l1 = nn.Linear(input_dim, 256)
        self.net2_l2 = nn.Linear(256, 256)
        self.net2_l3 = nn.Linear(256, 256)
        self.net2_l4 = nn.Linear(256, 1)

    
    def forward(self, state, goal, action):
        x_in = torch.cat([state, goal, action], 1)

        if self.half_cheetah:
            x_in = torch.cat([state, action], 1)

        # network 1
        x1 = torch.relu(self.net1_l1(x_in))
        x1 = torch.relu(self.net1_l2(x1))
        x1 = torch.relu(self.net1_l3(x1))
        x1 = self.net1_l4(x1)

        # network 2
        x2 = torch.relu(self.net2_l1(x_in))
        x2 = torch.relu(self.net2_l2(x2))
        x2 = torch.relu(self.net2_l3(x2))
        x2 = self.net2_l4(x2)

        return x1, x2