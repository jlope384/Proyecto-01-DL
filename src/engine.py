"""Entrenamiento y evaluación del MLP con early stopping sobre RMSE de validación."""

import copy

import numpy as np
import torch
import torch.nn as nn


def rmse_from_loader(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu())
            targets.append(yb)
    preds = torch.cat(preds).numpy().ravel()
    targets = torch.cat(targets).numpy().ravel()
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


def fit(model, train_loader, val_loader, epochs=200, lr=1e-3, weight_decay=0.0,
        patience=20, device="cpu", verbose=False):
    """Entrena con early stopping sobre val RMSE. Retorna (mejor_modelo, historial)."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    history = {"train_rmse": [], "val_rmse": []}
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        train_rmse = rmse_from_loader(model, train_loader, device)
        val_rmse = rmse_from_loader(model, val_loader, device)
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose and epoch % 10 == 0:
            print(f"epoch {epoch:3d} | train_rmse={train_rmse:.4f} | val_rmse={val_rmse:.4f}")

        if bad_epochs >= patience:
            if verbose:
                print(f"Early stopping en epoch {epoch} (best_val_rmse={best_val:.4f})")
            break

    model.load_state_dict(best_state)
    return model, history
