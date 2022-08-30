import numpy as np

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels, in_window, n_outputs):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=0),
            nn.ReLU(inplace=True),
        )
        # self.dense = nn.Sequential(
        #     nn.Linear(8 * (in_window - 2)**2, 16),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(16, n_outputs),
        # )
        self.dense = nn.Linear(8 * (in_window - 2)**2, n_outputs)

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view([x.shape[0], -1])
        x = self.dense(x)
        return x


def get_vector(model):
    state = model.state_dict()
    keys = sorted(list(state.keys()))
    vector = torch.cat([state[key].flatten() for key in keys], 0)
    return vector.numpy()

def set_params(model, vector):
    state = model.state_dict()
    keys = sorted(list(state.keys()))
    count = 0
    for key in keys:
        shape = state[key].shape
        dtype = state[key].dtype
        size = np.prod(shape)
        state[key] = torch.tensor(vector[count: count + size], dtype=dtype).view(shape)
        count = count + size
    model.load_state_dict(state)