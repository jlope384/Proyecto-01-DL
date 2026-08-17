# Proyecto 1: Competencia de Modelación — CC3092 Deep Learning y Sistemas Inteligentes

Autor: Javier Lopez
Fecha: 17/08/2026

## 2.1 Análisis exploratorio de datos (EDA)

_Ver `notebooks/01_eda.ipynb` para el detalle completo (estadísticas, gráficas)._

**Dimensiones y variable objetivo.** El dataset (`data/train.csv`, Ames
Housing) tiene 1168 filas y 81 columnas: 38 numéricas y 43 categóricas
(varias numéricas son en realidad categóricas codificadas como enteros, p.ej.
`MSSubClass`). La variable objetivo es `SalePrice` (precio de venta en USD),
continua.

**Estadísticas descriptivas del target.** Media $181,442, mediana $165,000,
desviación estándar $77,264, rango [$34,900, $745,000]. `SalePrice` tiene
sesgo positivo (skewness = 1.74); aplicar `log1p` reduce el sesgo a 0.12
(mucho más simétrico) — esto motivó probar `log1p(SalePrice)` como target
alternativo en la metodología.

**Valores nulos.** 19 columnas tienen NA. La mayoría (`PoolQC`,
`MiscFeature`, `Alley`, `Fence`, `FireplaceQu`, `GarageType/Finish/Qual/Cond`,
`BsmtQual/Cond/Exposure/FinType1/FinType2`, `MasVnrType`, y las numéricas
`GarageYrBlt`/`MasVnrArea`) representan **ausencia del feature**, no un dato
faltante real (según el data dictionary de Ames Housing) — se rellenan con
`"None"`/`0` explícito en vez de moda/media. `LotFrontage` (18% NA) y
`Electrical` (1 NA) sí son missing real y se imputan con mediana/moda dentro
del pipeline.

**Outliers.** Dos casas con `GrLivArea` > 4000 sq ft vendidas por debajo de
$200,000 — atípicas frente al patrón general (precio sube con el área). Se
evaluó experimentalmente el efecto de removerlas (iteración 7, ver 2.3).

**Correlación con el target (top numéricas):** `OverallQual` (0.79),
`GrLivArea` (0.70), `GarageCars` (0.64), `GarageArea` (0.62), `TotalBsmtSF`
(0.60), `1stFlrSF` (0.59), `FullBath` (0.55), `YearBuilt` (0.52). Entre
categóricas, `Neighborhood` y `KitchenQual` muestran diferencias claras de
mediana de `SalePrice` entre categorías.

**Decisiones de preprocesamiento derivadas del EDA** (implementadas en
`src/preprocessing.py`):
1. NA estructural -> `"None"`/`0` explícito (no moda/media).
2. `LotFrontage`/`Electrical` -> imputación mediana/moda (missing real).
3. Columnas de calidad ordinal (`ExterQual`, `KitchenQual`,
   `BsmtExposure`, etc.) -> mapeo a escala numérica ordinal en vez de one-hot.
4. Resto de categóricas nominales -> one-hot (`handle_unknown="ignore"`).
5. Numéricas y ordinales -> `StandardScaler` tras imputar.
6. Se documentan como iteraciones a validar (no decisiones de entrada):
   target `log1p` y remoción de los 2 outliers de `GrLivArea`.

## 2.2 Metodología de desarrollo

**Arquitecturas consideradas.** MLP totalmente conectado en PyTorch
(`src/model.py`), configurable en profundidad/ancho: se probaron `(64,)` y
`(128, 64)` neuronas por capa oculta, activación ReLU, con `Dropout` y
`BatchNorm1d` opcionales entre capas.

**División de datos.** Durante la fase de experimentación (2.3):
train/validation 80/20 (`random_state=42`), sin validación cruzada (dataset
pequeño y suficientemente representativo para una sola partición; ver
limitaciones en 2.4). Para el modelo final (`notebooks/03_final_model.ipynb`):
un 15% del dataset se aparta como held-out simulado (nunca se entrena con
él, `random_state=123`) para reportar un RMSE final honesto; dentro del 85%
restante se usa un split adicional 85/15 solo para decidir el número de
épocas vía early stopping, y luego se reentrena desde cero sobre el 100% de
ese 85% con el número de épocas ya decidido, para aprovechar todos los datos
disponibles antes de la competencia.

**Función de pérdida y optimizador.** `MSELoss`, optimizador `Adam`. Tasa de
aprendizaje 1e-3 (se probó 3e-4, ver iteración 7 de la ronda anterior de
experimentos). `weight_decay` (L2) explorado en 0 y 1e-4.

**Regularización.** `Dropout(0.2)` y `BatchNorm1d` en las capas ocultas,
`early stopping` sobre RMSE de validación (paciencia 25 épocas) — implementado
en `src/engine.py::fit`.

**Transformación del target.** Ver `src/target_transform.py`: soporte para
`log1p(SalePrice)` y/o estandarización (media/desviación del train) antes de
entrenar, con función inversa para reportar RMSE siempre en la escala
original de `SalePrice`. Esto no fue una decisión de entrada sino el
resultado de un hallazgo durante la experimentación (ver 2.3/2.4).

## 2.3 Resultados de iteraciones

Tabla completa en `experiments/experiments_log.csv`; curvas de entrenamiento
en `notebooks/02_experiments.ipynb`.

| Iteración | Cambios vs. anterior | Train RMSE | Val RMSE |
|---|---|---:|---:|
| iter1_baseline | MLP (64,), sin regularización, target crudo | 63,166 | 64,544 |
| iter2_wider_deeper | MLP (128,64), target crudo | 28,096 | 29,235 |
| iter3_regularized_FAILS | + dropout 0.2 + batchnorm, target crudo | 186,005 | 189,608 |
| iter4_target_scaling_FIX | igual que iter3 + target estandarizado | 12,528 | 27,618 |
| iter5_log_target | + target `log1p` (además de estandarizado) | 13,538 | 26,637 |
| iter6_weight_decay | + weight decay 1e-4 | 13,460 | 26,838 |
| iter7_remove_outliers | + remueve 2 outliers de `GrLivArea` | 11,456 | **20,454** |
| iter8_final | igual que iter7, más épocas/paciencia | 9,054 | 20,571 |

**Iteración 3 — hallazgo relevante:** agregar dropout + batchnorm sobre el
target crudo (`SalePrice` ~ $10^5$) hizo que el entrenamiento casi se
congelara (val RMSE 190k, peor que el baseline sin regularización). Tanto
train como val RMSE apenas se movían época a época. La causa no fue
"demasiada regularización": `BatchNorm` normaliza cada capa oculta a media 0
/ varianza 1, quitándole a la red el atajo de amplificar la escala del
target en capas intermedias; sin ese atajo, la única forma de alcanzar una
salida ~180,000 es a través de los pesos de la capa final, y Adam mueve cada
parámetro a un ritmo acotado por la tasa de aprendizaje por paso — alcanzar
esa escala así habría tomado muchísimas más épocas de las entrenadas. La
iteración 4 confirma el diagnóstico: misma arquitectura exacta, único cambio
es estandarizar el target antes de entrenar, y el val RMSE baja de 189,608 a
27,618.

## 2.4 Discusión de resultados

- **Capacidad del modelo (iter1 -> iter2):** pasar de una capa oculta (64) a
  dos capas (128, 64) reduce el val RMSE de 64,544 a 29,235 — el modelo más
  chico estaba en underfitting claro (ni siquiera ajustaba bien el train).
- **Escala del target (iter2/3 -> iter4):** el hallazgo más importante del
  proceso. No es un tema menor de implementación: para cualquier red con
  BatchNorm, no normalizar el target puede llevar a la falsa conclusión de
  que "la regularización no sirve" cuando en realidad el optimizador nunca
  llegó a converger.
- **log1p vs. estandarizado (iter4 -> iter5):** mejora modesta (27,618 ->
  26,637). Consistente con el EDA: el target ya estaba siendo estandarizado
  en iter4, así que gran parte del beneficio de `log1p` (reducir sesgo) se
  solapa con lo que la estandarización sola ya corrige parcialmente.
- **Weight decay (iter5 -> iter6):** prácticamente neutro (26,637 -> 26,838),
  sugiere que el dropout + early stopping ya estaban controlando el
  overfitting razonablemente bien en este dataset pequeño.
- **Remover outliers (iter6 -> iter7):** mejora notable (26,838 -> 20,454),
  el mayor salto después de la corrección de escala. Dos casas con
  `GrLivArea` muy alta y precio bajo distorsionaban el ajuste.
- **Más épocas (iter7 -> iter8):** val RMSE prácticamente igual (20,454 vs
  20,571) con más del doble de épocas — el modelo ya había convergido en
  iter7; entrenar más no ayuda y roza el overfitting (train RMSE sigue
  bajando mientras val se estanca/empeora levemente).
- **Análisis de errores (held-out simulado, ver `03_final_model.ipynb`):**
  gráficas de predicho-vs-real y residuos, y tabla de los 10 errores
  absolutos más grandes por `Neighborhood`/`OverallQual`/`GrLivArea` —
  _completar con la lectura de esas figuras: ¿el error crece en casas caras
  (heterocedasticidad)? ¿se concentra en algún vecindario?_
- **Limitaciones:** dataset pequeño (~1200 filas) para un MLP, sin
  validación cruzada durante la experimentación (un solo split), y el
  held-out usado aquí es simulado, no el real de competencia — el RMSE final
  reportado en 2.5 puede diferir del que se obtenga el día de la
  presentación.
- **Trade-off complejidad vs. generalización:** el salto de mayor impacto no
  vino de más capacidad ni de más regularización explícita, sino de tratar
  correctamente la escala del target y los outliers — evidencia de que en
  este problema el techo estaba más en el preprocesamiento/target que en la
  arquitectura en sí.

## 2.5 Conclusiones

- **Desempeño final:** RMSE de **$18,264.76** sobre el held-out simulado
  (15% de `data/train.csv` nunca visto durante el entrenamiento del modelo
  final), equivalente a ~10% del `SalePrice` medio del dataset — resultado
  razonable para un MLP sobre Ames Housing, aunque el número que cuenta para
  la nota de competencia se mide sobre el dataset held-out real entregado el
  17 de agosto.
- **Aprendizajes técnicos:** (1) nunca entrenar una red de regresión con el
  target en su escala cruda, especialmente si hay BatchNorm — estandarizar
  el target es tan importante como estandarizar los features; (2) los NA que
  representan "ausencia de feature" deben tratarse distinto de missing real;
  (3) revisar outliers evidentes en el EDA (`GrLivArea`) puede pesar más que
  ajustar hiperparámetros finos.
- **Aprendizajes metodológicos:** llevar un log de iteraciones desde el
  inicio (`experiments/experiments_log.csv`) permitió detectar y explicar el
  fallo de iter3 en vez de descartarlo como "mala suerte" — ese fue el
  hallazgo más valioso del proyecto.
- **Mejoras futuras:** validación cruzada (k-fold) en vez de un solo split
  para estimar varianza del RMSE; imputar `LotFrontage` por mediana dentro
  de cada `Neighborhood` en vez de global; probar target encoding para
  categóricas de alta cardinalidad (`Neighborhood`); búsqueda más
  sistemática de hiperparámetros (grid/random search) en vez de barrido
  secuencial manual.

## 2.6 Enlace al repositorio de GitHub

https://github.com/jlope384/Proyecto-01-DL
