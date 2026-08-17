"""Conversión de arrays numpy (ya preprocesados) a DataLoaders de PyTorch."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = False) -> DataLoader:
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)
