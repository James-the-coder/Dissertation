import torch.nn as nn
import numpy as np
import torch

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ICMmodel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ICMmodel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.feature_dim = 64

        # state encoder net
        self.encoder_net = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, self.feature_dim))
        )

        # forward dynamics network
        self.forward_net = nn.Sequential(
            layer_init(nn.Linear(self.feature_dim + self.action_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, self.feature_dim))
        )

        # inverse dynamics net
        self.inverse_net = nn.Sequential(
            layer_init(nn.Linear(self.feature_dim * 2, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, self.action_dim))
        )


    def encoder(self, state):
        encoded_state = self.encoder_net(state)
        return encoded_state

    
    def forward(self, state, action):
        state_encoding = self.encoder(state)
        input_tensor = torch.cat([state_encoding, action], 1)
        return self.forward_net(input_tensor)
    

    def inverse(self, state, next_state):
        state_encoding = self.encoder(state)
        next_state_encoding = self.encoder(next_state)
        input_tensor = torch.cat([state_encoding, next_state_encoding], 1)
        return self.inverse_net(input_tensor)