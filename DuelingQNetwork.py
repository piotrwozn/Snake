import torch
import torch.nn as nn
import torch.optim as optim


class DuelingQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuelingQNetwork, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(input_shape[0], 128),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        feature = self.feature(x)
        advantage = self.advantage(feature)
        value = self.value(feature)
        return value + advantage - advantage.mean()
