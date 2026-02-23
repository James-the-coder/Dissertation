import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, max_action):
        super(Actor, self).__init__()
        input_dim = state_dim + goal_dim

        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action


    def forward(self, state, goal):
        x = torch.cat([state, goal], 1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))

        # output mean and log std for action distribution
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2) # numerical stability

        return mean, log_std, self.max_action