# Implementation taken from
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py

import torch.nn as nn
import numpy as np
import torch

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class RNDModel(nn.Module):
    def __init__(self, input_size):
        super(RNDModel, self).__init__()

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
            layer_init(nn.Linear(self.input_size, 512)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, 512))
        )

        # freeze target network params
        for param in self.target_net.parameters():
            param.requires_grad = False


    def forward(self, state):
        target_feature = self.target_net(state)
        predict_feature = self.predictor_net(state)

        return predict_feature, target_feature