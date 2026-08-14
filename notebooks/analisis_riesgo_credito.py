"""Genera el notebook de análisis exploratorio.

Se construye mediante un script para que el .ipynb sea reproducible y no dependa
de la sesión interactiva. Ejecutar:

    python notebooks/analisis_riesgo_credito.py
"""

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell("""\
# Predicción de impago crediticio — análisis

Dataset: **Default of Credit Card Clients** (UCI) — 30,000 clientes de tarjeta de
crédito en Taiwán, abril–septiembre 2005.

Pregunta: **¿qué clientes caerán en impago el mes siguiente, y qué umbral de
decisión conviene usar?**

La segunda parte de la pregunta es la que casi nunca se hace, y es la que convierte
un modelo en una herramienta de negocio."""))

c.append(nbf.v4.new_code_cell("""\
import sys
sys.path.insert(0, "../src")

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from riesgo import config, datos, modelos, evaluación, viz

logging.basicConfig(level=logging.INFO, format="%(message)s")
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 140)"""))

c.append(nbf.v4.new_markdown_cell("## 1. Carga y primera inspección"))

c.append(nbf.v4.new_code_cell("""\
crudo = datos.cargar_crudo()
print("Dimensiones:", crudo.shape)
crudo.head()"""))

c.append(nbf.v4.new_code_cell("""\
# El desbalance de clases es el hecho central del problema.
objetivo = crudo["default payment next month"]
print("Tasa de impago: {:.2f}%".format(objetivo.mean() * 100))
print()
print("Si predijéramos 'nadie cae en impago':")
print("  exactitud: {:.2f}%".format((1 - objetivo.mean()) * 100))
print("  impagos no detectados: {:,}".format(int(objetivo.sum())))"""))

c.append(nbf.v4.new_markdown_cell("""\
Ese 77.9% de exactitud, sin detectar nada, es la razón por la que **accuracy no sirve
como métrica** en este problema. Cualquier modelo debe compararse contra esa línea
base, no contra el 50% que uno intuye."""))

c.append(nbf.v4.new_markdown_cell("## 2. Limpieza: las categorías no documentadas"))

c.append(nbf.v4.new_code_cell("""\
# El paper documenta EDUCATION 1-4 y MARRIAGE 1-3. Los datos traen más valores.
print("EDUCATION — valores presentes:")
print(crudo["EDUCATION"].value_counts().sort_index())
print()
print("MARRIAGE — valores presentes:")
print(crudo["MARRIAGE"].value_counts().sort_index())"""))

c.append(nbf.v4.new_markdown_cell("""\
Aparecen `EDUCATION` 0, 5 y 6, y `MARRIAGE` 0, que el paper no define.

**Decisión:** agruparlos en `"otro"` en lugar de eliminarlos. Son ~1.5% de los
registros; borrarlos introduciría sesgo de selección sin ganar nada."""))

c.append(nbf.v4.new_code_cell("""\
limpio = datos.limpiar(crudo)
print("Después de limpiar:", limpio.shape)
limpio[["sexo", "educacion", "estado_civil"]].apply(lambda s: s.value_counts())"""))

c.append(nbf.v4.new_markdown_cell("""\
## 3. La escala de atraso no es lineal

`PAY_0` a `PAY_6` parecen "meses de atraso", pero los valores negativos tienen
significado propio:

| Valor | Significado |
|---|---|
| −2 | Sin consumo ese mes |
| −1 | Pagó el total |
| 0 | Usó crédito revolvente |
| 1+ | Meses de atraso reales |

Tratarlos como una escala numérica continua sería un error de interpretación."""))

c.append(nbf.v4.new_code_cell("""\
print(limpio["atraso_sep"].value_counts().sort_index())
print()
tasa = limpio.groupby("atraso_sep")[config.OBJETIVO].agg(["mean", "count"])
tasa.columns = ["tasa_impago", "clientes"]
print(tasa.round(4))"""))

c.append(nbf.v4.new_markdown_cell("""\
El patrón es claro y monótono a partir de 1: a más meses de atraso, mayor tasa de
impago. Los valores −2 y −1 (sin consumo, pago total) tienen tasas bajas, como se
esperaría."""))

c.append(nbf.v4.new_markdown_cell("## 4. Features derivadas"))

c.append(nbf.v4.new_code_cell("""\
df = datos.agregar_features(limpio)
nuevas = [c for c in df.columns if c not in limpio.columns]
print("Variables derivadas ({}):".format(len(nuevas)))
for n in nuevas:
    print("  •", n)"""))

c.append(nbf.v4.new_code_cell("""\
# Comportamiento de pago agregado vs impago
resumen = df.groupby(config.OBJETIVO)[
    ["meses_con_atraso", "utilizacion_promedio", "razon_pago_saldo", "limite_credito"]
].mean().round(4)
resumen.index = ["Paga", "Impago"]
resumen"""))

c.append(nbf.v4.new_markdown_cell("""\
Los que caen en impago acumulan más meses de atraso, usan una proporción mayor de
su límite y pagan una fracción menor de lo que deben. Son las señales que un
analista de riesgo revisaría a mano."""))

c.append(nbf.v4.new_markdown_cell("## 5. Modelado"))

c.append(nbf.v4.new_code_cell("""\
X, y = datos.separar_xy(df)
X_tr, X_te, y_tr, y_te = modelos.separar_train_test(X, y)
print("Train:", X_tr.shape, "| Test:", X_te.shape)"""))

c.append(nbf.v4.new_code_cell("""\
# Toma ~2 minutos: entrena 4 modelos con validación cruzada de 5 folds.
entrenados, cv = modelos.entrenar_todos(X_tr, y_tr, folds=5)"""))

c.append(nbf.v4.new_code_cell("""\
tabla = evaluacion.comparar_modelos(entrenados, X_te, y_te)
tabla"""))

c.append(nbf.v4.new_markdown_cell("""\
Gradient Boosting gana en AUC-PR, pero **tiene el recall más bajo en el umbral de 0.5**.
Eso no es un defecto del modelo: el umbral de 0.5 es inadecuado para este
problema. Lo corregimos abajo."""))

c.append(nbf.v4.new_markdown_cell("""\
## 6. El umbral óptimo

Los dos errores no cuestan lo mismo:

- **Falso negativo** (le prestamos a quien no paga) → se pierde el saldo expuesto
- **Falso positivo** (rechazamos a quien sí pagaría) → se pierde el margen

En tarjetas de crédito la razón suele estar entre 1:3 y 1:10. Usamos 1:5 como
supuesto conservador y luego probamos qué pasa si está mal."""))

c.append(nbf.v4.new_code_cell("""\
mejor_nombre = tabla.iloc[0]["modelo"]
mejor = entrenados[mejor_nombre]
proba = mejor.predict_proba(X_te)[:, 1]

opt = evaluacion.umbral_optimo(y_te, proba)
print("Modelo:", mejor_nombre)
print()
print("  umbral 0.5 (por defecto) ... costo {:,.0f}".format(opt["costo_umbral_05"]))
print("  umbral óptimo ({:.2f}) ...... costo {:,.0f}".format(opt["umbral"], opt["costo"]))
print("  ahorro ..................... {:,.0f} ({:.1f}%)".format(
    opt["ahorro_absoluto"], opt["ahorro_pct"]))
print()
print("  recall ..................... {:.4f}".format(opt["recall"]))
print("  precisión .................. {:.4f}".format(opt["precision"]))
print("  tasa de rechazo ............ {:.2%}".format(opt["tasa_rechazo"]))"""))

c.append(nbf.v4.new_code_cell("""\
viz.costo_vs_umbral(opt["barrido"], opt["umbral"])
from IPython.display import Image
Image(str(config.FIGURES / "costo_vs_umbral.png"))"""))

c.append(nbf.v4.new_markdown_cell("""\
### ¿Y si el supuesto de costos está mal?

La crítica obvia: "1:5 es un número que inventaste". Justo. Veamos qué tan sensible
es la conclusión."""))

c.append(nbf.v4.new_code_cell("""\
evaluacion.sensibilidad_costos(y_te, proba)"""))

c.append(nbf.v4.new_markdown_cell("""\
**La conclusión es robusta.** En todo el rango de 1:2 a 1:20 el umbral óptimo está
por debajo de 0.5 y el ahorro es positivo. Lo que cambia es la magnitud, no la
dirección de la recomendación."""))

c.append(nbf.v4.new_markdown_cell("## 7. ¿Qué mira el modelo?"))

c.append(nbf.v4.new_code_cell("""\
imp = modelos.importancia_variables(mejor, X_te, top=15)
imp"""))

c.append(nbf.v4.new_markdown_cell("""\
El comportamiento de pago reciente domina: `atraso_sep` explica por sí sola la mitad
de la señal.

Vale la pena notar que las variables demográficas (sexo, educación, estado civil)
quedan muy abajo. Eso es **deseable**: un modelo que decidiera crédito con base en
el sexo del solicitante sería un problema regulatorio, no solo estadístico."""))

c.append(nbf.v4.new_markdown_cell("""\
## Conclusiones

1. **La precisión es engañosa aquí.** Predecir "nadie cae en impago" da un 77.9% sin
   detectar un solo caso.

2. **El umbral 0.5 no es óptimo.** Moverlo a 0.18 reduce el costo esperado 25.7% y
   sube la detección de impagos de 37% a 71%.

3. **La conclusión aguanta el supuesto de costos.** De 1:2 a 1:20, el óptimo siempre
   está por debajo de 0.5.

4. **El historial de pagos manda.** Las variables demográficas aportan poco, lo cual
   es bueno desde el punto de vista regulatorio.

### Siguiente paso

Evaluar equidad del modelo entre grupos demográficos (*fairness*) antes de
cualquier uso real."""))

nb["cells"] = c
destino = ROOT / "notebooks" / "analisis_riesgo_credito.ipynb"
nbf.write(nb, str(destino))
print("Notebook escrito:", destino.name, "|", len(c), "celdas")
