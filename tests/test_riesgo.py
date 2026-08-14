"""Pruebas de limpieza, features y evaluacion.

Ejecutar:  python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from riesgo import config, datos, evaluacion


@pytest.fixture
def df_crudo():
    """Muestra minima con la estructura del dataset original."""
    n = 6
    d = {
        "ID": range(1, n + 1),
        "limite_credito": [20000, 120000, 90000, 50000, 50000, 500000],
        "sexo": [2, 2, 2, 1, 1, 1],
        # 0, 5 y 6 son categorias no documentadas en el paper.
        "educacion": [2, 2, 0, 5, 6, 1],
        "estado_civil": [1, 2, 0, 1, 2, 3],
        "edad": [24, 26, 34, 37, 57, 29],
    }
    for c in config.MESES_ATRASO:
        d[c] = [2, -1, 0, 1, -2, 0]
    for c in config.MESES_SALDO:
        d[c] = [3913, 2682, 29239, 46990, 8617, 64400]
    for c in config.MESES_PAGO:
        d[c] = [0, 1000, 1518, 2000, 5000, 3000]
    d[config.OBJETIVO] = [1, 1, 0, 0, 0, 0]
    return pd.DataFrame(d)


# --- Limpieza -----------------------------------------------------------

def test_limpiar_mapea_categorias_no_documentadas(df_crudo):
    """EDUCATION 0/5/6 y MARRIAGE 0 deben caer en 'otro', no perderse."""
    out = datos.limpiar(df_crudo)

    assert set(out["educacion"]).issubset(
        {"posgrado", "universidad", "preparatoria", "otro"})
    assert (out["educacion"] == "otro").sum() == 3   # los valores 0, 5 y 6
    assert (out["estado_civil"] == "otro").sum() == 2  # el 0 y el 3


def test_limpiar_no_pierde_filas(df_crudo):
    """Las categorias raras se agrupan; no se eliminan registros."""
    out = datos.limpiar(df_crudo)
    assert len(out) == len(df_crudo)


def test_limpiar_elimina_columna_id(df_crudo):
    out = datos.limpiar(df_crudo)
    assert "ID" not in out.columns


def test_limpiar_quita_duplicados(df_crudo):
    doble = pd.concat([df_crudo, df_crudo.iloc[[0]]], ignore_index=True)
    out = datos.limpiar(doble)
    assert len(out) == len(df_crudo)


# --- Features -----------------------------------------------------------

def test_features_cuenta_meses_con_atraso(df_crudo):
    """Solo valores >= 1 cuentan como atraso real: -2/-1/0 no lo son."""
    out = datos.agregar_features(datos.limpiar(df_crudo))

    assert out.loc[0, "meses_con_atraso"] == 6   # todos en 2
    assert out.loc[1, "meses_con_atraso"] == 0   # todos en -1 (pago total)
    assert out.loc[4, "meses_con_atraso"] == 0   # todos en -2 (sin consumo)


def test_features_utilizacion_en_rango(df_crudo):
    out = datos.agregar_features(datos.limpiar(df_crudo))
    util = out["utilizacion_sep"].dropna()
    assert (util >= 0).all()


def test_features_sin_infinitos(df_crudo):
    """La division por limite o saldo cero no debe dejar inf."""
    d = df_crudo.copy()
    d.loc[0, "limite_credito"] = 0
    for c in config.MESES_SALDO:
        d.loc[1, c] = 0

    out = datos.agregar_features(datos.limpiar(d))
    numericas = out.select_dtypes(include=[np.number])
    assert not np.isinf(numericas.to_numpy()).any()


def test_features_agrega_columnas(df_crudo):
    limpio = datos.limpiar(df_crudo)
    out = datos.agregar_features(limpio)
    assert out.shape[1] > limpio.shape[1]
    for col in ["meses_con_atraso", "utilizacion_sep", "razon_pago_saldo"]:
        assert col in out.columns


# --- Evaluacion ---------------------------------------------------------

def test_linea_base_trivial():
    """Con 22% de positivos, predecir 'todo cero' da 78% de exactitud."""
    y = pd.Series([0] * 78 + [1] * 22)
    base = evaluacion.linea_base_trivial(y)

    assert base["exactitud"] == pytest.approx(0.78)
    assert base["recall"] == 0.0
    assert base["impagos_no_detectados"] == 22


def test_costo_penaliza_mas_los_falsos_negativos():
    y_true = np.array([0, 0, 1, 1])
    solo_fn = np.array([0, 0, 0, 1])   # 1 falso negativo
    solo_fp = np.array([0, 1, 1, 1])   # 1 falso positivo

    c_fn = evaluacion.costo_total(y_true, solo_fn, costo_fn=5.0, costo_fp=1.0)
    c_fp = evaluacion.costo_total(y_true, solo_fp, costo_fn=5.0, costo_fp=1.0)

    assert c_fn == 5.0
    assert c_fp == 1.0
    assert c_fn > c_fp


def test_umbral_optimo_no_es_peor_que_05():
    """Por construccion, el optimo nunca cuesta mas que el umbral 0.5."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    proba = np.clip(y * 0.35 + rng.normal(0.35, 0.2, 400), 0.01, 0.99)

    res = evaluacion.umbral_optimo(y, proba)
    assert res["costo"] <= res["costo_umbral_05"]
    assert 0 < res["umbral"] < 1


def test_metricas_estructura():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    m = evaluacion.metricas(y_true, y_pred)

    assert m["verdaderos_positivos"] == 1
    assert m["falsos_positivos"] == 1
    assert m["falsos_negativos"] == 1
    assert m["verdaderos_negativos"] == 1
    assert m["recall"] == pytest.approx(0.5)


def test_sensibilidad_recorre_razones():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    proba = np.clip(y * 0.3 + rng.normal(0.35, 0.2, 300), 0.01, 0.99)

    sens = evaluacion.sensibilidad_costos(y, proba, razones=(2, 5, 10))
    assert len(sens) == 3
    # A mayor costo del falso negativo, el umbral optimo no deberia subir.
    assert sens["umbral_optimo"].iloc[-1] <= sens["umbral_optimo"].iloc[0]
