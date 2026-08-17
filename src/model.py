"""MLP para regresión de SalePrice."""

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(128, 64), dropout=0.0, batchnorm=False):
        super().__init__()
        layers = []
        prev = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            if batchnorm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = size
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
