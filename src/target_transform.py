"""Transformación (y su inversa) del target SalePrice para entrenamiento.

Entrenar una red con el target en su escala cruda (SalePrice ~ 10^5) es una
mala práctica conocida: la salida de la red parte cerca de 0 y el modelo debe
"levantar" la escala completa a través de los pesos/bias, lo cual con Adam
ocurre a un ritmo acotado por la tasa de aprendizaje por paso. Esto se vuelve
crítico con BatchNorm, que normaliza cada capa oculta a media 0 / var 1 y le
quita a la red el atajo de amplificar la escala en capas intermedias -> el
entrenamiento se estanca (ver notebooks/02_experiments.ipynb, iteración 3).

Este módulo estandariza (y opcionalmente aplica log1p a) el target antes de
entrenar, y deshace la transformación sobre las predicciones.
"""

import numpy as np


def fit_target_transform(y_raw, log_target=False, scale_target=False):
    y = np.log1p(y_raw) if log_target else np.asarray(y_raw, dtype="float64")
    mean = float(y.mean()) if scale_target else 0.0
    std = float(y.std()) if scale_target else 1.0
    y_t = (y - mean) / std
    stats = {"log_target": log_target, "scale_target": scale_target, "mean": mean, "std": std}
    return y_t.astype("float32"), stats


def apply_target_transform(y_raw, stats):
    y = np.log1p(y_raw) if stats["log_target"] else np.asarray(y_raw, dtype="float64")
    y_t = (y - stats["mean"]) / stats["std"]
    return y_t.astype("float32")


def inverse_transform_target(y_pred, stats):
    y = y_pred * stats["std"] + stats["mean"]
    return np.expm1(y) if stats["log_target"] else y
