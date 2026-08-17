"""Pipeline de preprocesamiento para el dataset Ames Housing (target: SalePrice).

Decisiones tomadas durante el EDA (ver notebooks/01_eda.ipynb):

- Varias columnas categóricas tienen NaN cuando el feature simplemente no
  existe en esa casa (p.ej. PoolQC = NaN significa "sin piscina", no un dato
  faltante). Esas se rellenan con la categoría explícita "None" en vez de con
  la moda.
- GarageYrBlt y MasVnrArea son numéricas con el mismo tipo de NaN
  estructural -> se rellenan con 0.
- Las columnas de calidad (Ex/Gd/TA/Fa/Po y similares) tienen una relación de
  orden real, así que se mapean a una escala ordinal en vez de one-hot.
- El resto de categóricas nominales van a one-hot encoding.
- Las numéricas restantes (incluyendo LotFrontage, que sí tiene missing real)
  se imputan con la mediana y se escalan con StandardScaler.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "SalePrice"
ID_COL = "Id"

NONE_CATEGORICAL = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]

ZERO_NUMERIC = ["GarageYrBlt", "MasVnrArea"]

QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]

BSMT_EXPOSURE_MAP = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
BSMT_FINTYPE_MAP = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
GARAGE_FINISH_MAP = {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}

ORDINAL_MAPS = {
    **{c: QUALITY_MAP for c in QUALITY_COLS},
    "BsmtExposure": BSMT_EXPOSURE_MAP,
    "BsmtFinType1": BSMT_FINTYPE_MAP,
    "BsmtFinType2": BSMT_FINTYPE_MAP,
    "GarageFinish": GARAGE_FINISH_MAP,
}


def fill_structural_na(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena los NaN que representan 'ausencia de feature', no missing real."""
    df = df.copy()
    for col in NONE_CATEGORICAL:
        if col in df:
            df[col] = df[col].fillna("None")
    for col in ZERO_NUMERIC:
        if col in df:
            df[col] = df[col].fillna(0)
    return df


def apply_ordinal_maps(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas de calidad tipo Ex/Gd/TA/Fa/Po (y similares) a ordinal numérico."""
    df = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        if col in df:
            df[col] = df[col].map(mapping)
    return df


def prepare_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza que debe aplicarse ANTES del ColumnTransformer."""
    return apply_ordinal_maps(fill_structural_na(df))


def get_feature_lists(df: pd.DataFrame):
    """Separa columnas (ya pasadas por prepare_raw) en numéricas, ordinales y nominales."""
    cols = [c for c in df.columns if c not in (ID_COL, TARGET)]
    ordinal_cols = [c for c in ORDINAL_MAPS if c in cols]
    remaining = [c for c in cols if c not in ordinal_cols]
    numeric_cols = [c for c in remaining if pd.api.types.is_numeric_dtype(df[c])]
    nominal_cols = [c for c in remaining if c not in numeric_cols]
    return numeric_cols, ordinal_cols, nominal_cols


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Construye el ColumnTransformer. `df` debe haber pasado ya por prepare_raw."""
    numeric_cols, ordinal_cols, nominal_cols = get_feature_lists(df)

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    ordinal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    nominal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("ord", ordinal_pipe, ordinal_cols),
        ("nom", nominal_pipe, nominal_cols),
    ])
