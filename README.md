# Proyecto 1: Competencia de Modelación (CC3092)

MLP en PyTorch para predecir `SalePrice` sobre el dataset Ames Housing.
Modelo final: ensamble de 5 MLPs (bagging). RMSE final sobre held-out
simulado: **$16,755.79** (ver `docs/informe.md`).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Estructura

```
data/                   train.csv (y test.csv el día de la competencia)
notebooks/
  01_eda.ipynb           análisis exploratorio y decisiones de preprocesamiento
  02_experiments.ipynb   barrido de iteraciones, tabla de resultados, curvas
  03_final_model.ipynb   entrena el modelo final y guarda artifacts/
src/
  preprocessing.py       limpieza de NA estructurales, mapeo ordinal, ColumnTransformer
  dataset.py              numpy -> DataLoader de PyTorch
  model.py                clase MLP configurable
  engine.py               loop de entrenamiento con early stopping
  predict.py               script para generar predicciones sobre un CSV nuevo
experiments/
  experiments_log.csv     tabla de iteraciones (train/val RMSE, cambios, notas)
artifacts/                 preprocessor.joblib, model_0..model_4.pt (ensamble), config.json
docs/
  informe.md               trabajo escrito (secciones 2.1-2.6 del enunciado)
```

`src/target_transform.py` maneja la transformación del target (`log1p` y/o
estandarización) y su inversa — necesario porque entrenar sobre `SalePrice`
crudo con BatchNorm hace que el entrenamiento casi no converja (ver
iteración 3 en `docs/informe.md`, sección 2.3).

## Reproducir resultados

1. Correr `notebooks/01_eda.ipynb` de principio a fin.
2. Correr `notebooks/02_experiments.ipynb` para reproducir el barrido de
   iteraciones (actualiza `experiments/experiments_log.csv`).
3. Correr `notebooks/03_final_model.ipynb` para entrenar el modelo final y
   generar `artifacts/`.
4. Generar predicciones sobre un dataset nuevo (p.ej. el held-out del día de
   la competencia):

   ```bash
   python src/predict.py --input data/test.csv --output predictions.csv
   ```

   Si el CSV de entrada incluye la columna `SalePrice`, el script imprime el
   RMSE calculado contra los valores reales. `data/holdout_simulated.csv`
   (generado por el notebook 03) sirve para probar el script antes del día
   de la competencia.

## Enlace al repositorio

https://github.com/jlope384/Proyecto-01-DL
