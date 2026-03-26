# Implementation taken from
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py

import torch.nn as nn
import numpy as np
import torch

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# New Xavier Initialization
def layer_init_xavier(layer, bias_const=0.0):
    # Calculate gain for LeakyReLU to prevent vanishing gradients
    gain = torch.nn.init.calculate_gain('leaky_relu') 
    
    # Apply Xavier Uniform initialization
    torch.nn.init.xavier_uniform_(layer.weight, gain=gain)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class RNDModel(nn.Module):
    def __init__(self, input_size, masked=False):
        super(RNDModel, self).__init__()

        self.masked = masked

        if self.masked:
            self.input_size = 15
        else:
            self.input_size = input_size

        # predictor network
        self.predictor_net = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 512)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, 512))
        )

        # target network
        self.target_net = nn.Sequential(
            layer_init_xavier(nn.Linear(self.input_size, 512)),
            nn.LeakyReLU(),
            layer_init_xavier(nn.Linear(512, 512)),
            nn.LeakyReLU(),
            layer_init_xavier(nn.Linear(512, 512)),
            nn.LeakyReLU(),
            layer_init_xavier(nn.Linear(512, 512))
        )

        # freeze target network params
        for param in self.target_net.parameters():
            param.requires_grad = False


    def forward(self, state):
        if self.masked:
            # use only block data + relative arm and block data
            feature_idxs = [3,4,5,6,7,8,11,12,13,14,15,16,17,18,19]
            state = state[..., feature_idxs]

        target_feature = self.target_net(state)
        predict_feature = self.predictor_net(state)

        return predict_feature, target_feature