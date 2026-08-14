# Predicción de impago crediticio

Modelo de clasificación que estima la probabilidad de que un cliente de tarjeta de
crédito caiga en impago, y que además calcula qué umbral de decisión conviene usar
según lo que cuesta cada tipo de error.

Dataset: [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
(UCI), 30,000 clientes reales de Taiwán, abril a septiembre de 2005.

---

## El problema con la mayoría de los modelos de crédito

El 22.1% de los clientes cae en impago. Eso significa que un modelo que prediga
"nadie cae en impago", sin detectar un solo caso, alcanza 77.9% de exactitud.

Por eso aquí no reporto accuracy como métrica principal, sino AUC-PR, recall y el
costo esperado en unidades de negocio.

```
Modelo trivial ("nadie cae en impago"):
  exactitud .................. 0.7788
  recall ..................... 0.0000
  impagos no detectados ...... 1,659
```

---

## Resultado principal

El umbral de decisión por defecto (0.5) no es el óptimo. Los dos errores no cuestan
lo mismo: aprobar a alguien que no paga cuesta el saldo expuesto, mientras que
rechazar a alguien que sí habría pagado cuesta solo el margen.

Con una razón de costos conservadora de 1:5:

| Umbral | Costo total | Recall | Precisión |
|---|---|---|---|
| 0.50 (por defecto) | 5,548 | 0.369 | 0.662 |
| 0.18 (óptimo) | 4,122 | 0.715 | 0.403 |

Mover el umbral reduce el costo 25.7% y casi duplica la detección de impagos, de
37% a 71%, a cambio de más rechazos preventivos.

![Costo vs umbral](reports/figures/costo_vs_umbral.png)

### ¿Y si el supuesto de costos está mal?

La razón 1:5 es un supuesto mío, no un dato. El análisis de sensibilidad la varía
de 1:2 a 1:20:

| Razón de costos | Umbral óptimo | Recall | Precisión | Ahorro vs 0.5 |
|---|---|---|---|---|
| 1:2 | 0.32 | 0.535 | 0.555 | 6.3% |
| 1:3 | 0.26 | 0.593 | 0.508 | 13.8% |
| 1:5 | 0.18 | 0.715 | 0.403 | 25.7% |
| 1:10 | 0.08 | 0.952 | 0.263 | 51.6% |
| 1:20 | 0.07 | 0.972 | 0.252 | 73.1% |

En todo el rango evaluado el umbral óptimo queda por debajo de 0.5 y el ahorro sigue
siendo positivo. Lo que cambia es cuánto se ahorra, no si conviene mover el umbral.

---

## Comparación de modelos

| Modelo | AUC-PR | ROC-AUC | Recall | Precisión | Exactitud |
|---|---|---|---|---|---|
| Gradient Boosting | 0.563 | 0.782 | 0.369 | 0.662 | 0.819 |
| Random Forest | 0.560 | 0.780 | 0.603 | 0.487 | 0.772 |
| Árbol de decisión | 0.533 | 0.765 | 0.623 | 0.454 | 0.751 |
| Regresión logística | 0.504 | 0.754 | 0.610 | 0.439 | 0.741 |

![Comparación](reports/figures/comparacion_modelos.png)

Gradient Boosting gana en AUC-PR pero tiene el recall más bajo al umbral 0.5. No es
un defecto del modelo: ese umbral simplemente no le sirve. Al reoptimizarlo su recall
sube a 0.715. Por eso comparar modelos solo al umbral por defecto lleva a
conclusiones equivocadas.

![Curva PR](reports/figures/curvas_precision_recall.png)

---

## Variables más predictivas

![Importancia](reports/figures/importancia_variables.png)

| Variable | Qué mide | Importancia |
|---|---|---|
| `atraso_sep` | Estatus de pago del mes más reciente | 0.508 |
| `meses_con_atraso` | Cuántos de los 6 meses tuvo atraso | 0.121 |
| `atraso_maximo` | Peor atraso del periodo | 0.076 |
| `atraso_promedio` | Atraso promedio | 0.044 |
| `utilizacion_maxima` | Porcentaje máximo del límite usado | 0.031 |

El comportamiento de pago reciente domina: `atraso_sep` por sí sola explica la mitad
de la señal. Sexo, educación y estado civil quedan muy abajo, y eso me parece bien.
Un modelo que decidiera a quién darle crédito según su sexo tendría un problema
regulatorio, no solo estadístico.

---

## Decisiones metodológicas

Uso AUC-PR como métrica principal, no ROC-AUC ni accuracy. Con 22% de positivos, la
curva precision-recall describe mucho mejor qué tan bien va el modelo sobre la clase
que importa. La ROC puede verse decente mientras el modelo falla justo en los casos
que quieres atrapar.

Las categorías no documentadas van a "otro" en lugar de eliminarse. El paper original
define `EDUCATION` de 1 a 4 y `MARRIAGE` de 1 a 3, pero los datos traen ceros y
valores extra que nadie explica. Son alrededor del 1.5% de los registros y borrarlos
metería sesgo de selección.

La escala de atraso no es lineal, aunque lo parezca. Los valores −2, −1 y 0 significan
"sin consumo", "pago total" y "crédito revolvente". Solo de 1 en adelante hay atraso
real. Conservo el valor original y derivo aparte una variable de atraso efectivo.

El escalado y la imputación van dentro del Pipeline de sklearn, no antes. Si los
ajustara sobre el dataset completo antes de separar train y test, información del
test se filtraría al modelo y las métricas saldrían infladas.

Todos los modelos usan `class_weight="balanced"`. Sin eso aprenden a predecir "paga"
casi siempre, porque el 78% de los casos lo son.

La validación cruzada es estratificada de 5 folds sobre el conjunto de entrenamiento.
El de prueba, 25% de los datos, lo toco una sola vez al final.

---

## Instalación y uso

```bash
git clone https://github.com/MoncerratAngeles/riesgo-credito-ml.git
cd riesgo-credito-ml

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

PYTHONPATH=src python -m riesgo.pipeline
```

El dataset se descarga solo la primera vez desde UCI. La ejecución completa toma
~2 minutos.

Opciones:

```bash
python -m riesgo.pipeline --folds 3        # validación cruzada más rápida
python -m riesgo.pipeline --sin-graficas
python -m riesgo.pipeline -v
```

Pruebas:

```bash
python -m pytest tests/ -v                 # 13 pruebas
```

---

## Estructura

```
riesgo-credito-ml/
├── src/riesgo/
│   ├── config.py       Rutas, diccionario de variables, supuestos de costo
│   ├── datos.py        Descarga, limpieza y features derivadas
│   ├── modelos.py      Pipelines de sklearn y validación cruzada
│   ├── evaluacion.py   Métricas, análisis de costos y umbral óptimo
│   ├── viz.py          Figuras
│   └── pipeline.py     Orquestador con CLI
├── tests/              13 pruebas unitarias
├── notebooks/          Análisis exploratorio narrado
├── reports/            Tablas de resultados y figuras
└── models/             Modelo entrenado (.joblib)
```

## Salidas

| Archivo | Contenido |
|---|---|
| `reports/resumen.json` | Métricas principales del mejor modelo |
| `reports/comparacion_modelos.csv` | Tabla comparativa completa |
| `reports/barrido_umbrales.csv` | Costo y métricas en 101 umbrales |
| `reports/sensibilidad_costos.csv` | Umbral óptimo según la razón de costos |
| `reports/importancia_variables.csv` | Ranking de variables |
| `models/modelo_final.joblib` | Modelo entrenado, listo para predecir |

## Stack

Python · scikit-learn · Pandas · NumPy · Matplotlib · pytest

## Limitaciones

Los datos son de Taiwán en 2005, así que los patrones concretos no se trasladan al
mercado mexicano de hoy. La metodología sí.

La razón de costos 1:5 la elegí yo. Es un supuesto razonable para tarjetas de
crédito, pero ninguna institución me dio ese número. Por eso incluí el análisis de
sensibilidad.

Falta evaluar si el modelo trata distinto a unos grupos demográficos que a otros.
Es lo que haría antes de proponerlo para cualquier uso real.

## Fuente

Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for
the predictive accuracy of probability of default of credit card clients*.
Expert Systems with Applications, 36(2), 2473–2480.

## Licencia

MIT
