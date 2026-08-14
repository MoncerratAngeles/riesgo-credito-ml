"""Configuracion del proyecto de riesgo crediticio.

Dataset: Default of Credit Card Clients (UCI, Yeh & Lien 2009).
30,000 clientes de tarjeta de credito en Taiwan, abril-septiembre 2005.
Variable objetivo: si el cliente cayo en impago el mes siguiente (octubre 2005).
"""

from pathlib import Path

# --- Rutas --------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for _d in (DATA_RAW, DATA_PROCESSED, MODELS, REPORTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# --- Fuente -------------------------------------------------------------
URL_DATASET = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)
ARCHIVO_CRUDO = DATA_RAW / "default_credit_card_clients.xls"
ARCHIVO_LIMPIO = DATA_PROCESSED / "credito_limpio.csv"

OBJETIVO = "impago"
SEMILLA = 42
TEST_SIZE = 0.25

# --- Diccionario de variables -------------------------------------------
# El dataset original usa nombres crípticos. Se renombran a algo legible.
RENOMBRES = {
    "LIMIT_BAL": "limite_credito",
    "SEX": "sexo",
    "EDUCATION": "educacion",
    "MARRIAGE": "estado_civil",
    "AGE": "edad",
    "PAY_0": "atraso_sep", "PAY_2": "atraso_ago", "PAY_3": "atraso_jul",
    "PAY_4": "atraso_jun", "PAY_5": "atraso_may", "PAY_6": "atraso_abr",
    "BILL_AMT1": "saldo_sep", "BILL_AMT2": "saldo_ago", "BILL_AMT3": "saldo_jul",
    "BILL_AMT4": "saldo_jun", "BILL_AMT5": "saldo_may", "BILL_AMT6": "saldo_abr",
    "PAY_AMT1": "pago_sep", "PAY_AMT2": "pago_ago", "PAY_AMT3": "pago_jul",
    "PAY_AMT4": "pago_jun", "PAY_AMT5": "pago_may", "PAY_AMT6": "pago_abr",
    "default payment next month": OBJETIVO,
}

MESES_ATRASO = ["atraso_sep", "atraso_ago", "atraso_jul",
                "atraso_jun", "atraso_may", "atraso_abr"]
MESES_SALDO = ["saldo_sep", "saldo_ago", "saldo_jul",
               "saldo_jun", "saldo_may", "saldo_abr"]
MESES_PAGO = ["pago_sep", "pago_ago", "pago_jul",
              "pago_jun", "pago_may", "pago_abr"]

# --- Categorias no documentadas -----------------------------------------
# El paper documenta EDUCATION 1-4 y MARRIAGE 1-3, pero los datos traen
# ceros y valores extra sin significado declarado. Se agrupan en "otro"
# en vez de eliminarse: son ~1.5% de los registros y descartarlos sesgaria.
EDUCACION_MAPA = {
    1: "posgrado", 2: "universidad", 3: "preparatoria",
    4: "otro", 5: "otro", 6: "otro", 0: "otro",
}
ESTADO_CIVIL_MAPA = {
    1: "casado", 2: "soltero", 3: "otro", 0: "otro",
}
SEXO_MAPA = {1: "hombre", 2: "mujer"}

# --- Costos de negocio ---------------------------------------------------
# El punto central del proyecto: los dos errores NO cuestan lo mismo.
#
# Falso negativo (predecimos que paga, pero cae en impago):
#   se pierde el saldo expuesto. Es el error caro.
# Falso positivo (predecimos impago, pero si habria pagado):
#   se pierde el margen de ese cliente y su relacion comercial.
#
# La razon 1:5 es un supuesto conservador para la industria de tarjetas.
# El analisis de sensibilidad en el notebook la varia de 1:2 a 1:20.
COSTO_FALSO_NEGATIVO = 5.0
COSTO_FALSO_POSITIVO = 1.0
