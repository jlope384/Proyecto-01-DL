"""Genera predicciones de SalePrice sobre un CSV nuevo usando los artifacts
guardados por notebooks/03_final_model.ipynb (pesos + preprocessor + config).

Pensado para el día de la competencia: recibe el dataset held-out, aplica el
mismo pipeline de preprocesamiento que se usó en entrenamiento y calcula RMSE
si el CSV trae la columna SalePrice.

Uso:
    python src/predict.py --input data/train.csv --output predictions.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from model import MLP
from preprocessing import ID_COL, TARGET, prepare_raw
from target_transform import inverse_transform_target

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def load_artifacts():
    preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
    config = json.loads((ARTIFACTS_DIR / "config.json").read_text())

    model = MLP(
        input_dim=config["input_dim"],
        hidden_sizes=config["hidden_sizes"],
        dropout=config["dropout"],
        batchnorm=config["batchnorm"],
    )
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "model.pt", map_location="cpu"))
    model.eval()
    return preprocessor, model, config


def predict(df: pd.DataFrame) -> np.ndarray:
    preprocessor, model, config = load_artifacts()
    clean = prepare_raw(df)
    X = preprocessor.transform(clean)
    with torch.no_grad():
        preds = model(torch.as_tensor(X, dtype=torch.float32)).numpy().ravel()
    return inverse_transform_target(preds, config)


def main():
    parser = argparse.ArgumentParser(description="Genera predicciones de SalePrice.")
    parser.add_argument("--input", required=True, help="CSV con las mismas columnas que train.csv")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    preds = predict(df)

    out = pd.DataFrame({ID_COL: df[ID_COL], "SalePrice": preds})
    out.to_csv(args.output, index=False)
    print(f"Predicciones guardadas en {args.output}")

    if TARGET in df.columns:
        rmse = float(np.sqrt(np.mean((preds - df[TARGET].values) ** 2)))
        print(f"RMSE vs. {TARGET} real: {rmse:.2f}")


if __name__ == "__main__":
    main()
