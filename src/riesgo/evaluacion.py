"""Metricas y analisis de costos.

En credito la accuracy enganya. Con 22% de impagos, un modelo que prediga
"todos pagan" saca 78% sin detectar uno solo. Por eso aqui se reporta
recall, precision y AUC-PR, ademas del costo esperado.

El umbral tampoco se deja en 0.5. Se busca el que minimiza el costo total,
porque un falso negativo (prestarle a quien no paga) sale varias veces mas
caro que un falso positivo (rechazar a quien si habria pagado).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config

log = logging.getLogger(__name__)


def metricas(y_true, y_pred, y_proba=None):
    """Conjunto de metricas para clasificacion desbalanceada."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    out = {
        "exactitud": (tp + tn) / len(y_true),
        "exactitud_balanceada": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "especificidad": tn / (tn + fp) if (tn + fp) else 0.0,
        "verdaderos_negativos": int(tn),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_positivos": int(tp),
    }
    if y_proba is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
        out["auc_pr"] = average_precision_score(y_true, y_proba)
    return out


def linea_base_trivial(y_true):
    """El modelo tonto: predice que nadie cae en impago.

    Va explicito en el reporte porque es la referencia honesta. Un modelo
    que no le gane en recall no esta aportando nada.
    """
    y_pred = np.zeros(len(y_true), dtype=int)
    return {
        "exactitud": float((y_pred == y_true).mean()),
        "recall": 0.0,
        "precision": 0.0,
        "impagos_no_detectados": int(np.sum(y_true)),
    }


def costo_total(y_true, y_pred, costo_fn=None, costo_fp=None):
    """Costo esperado en unidades de negocio.

    costo_fn: costo de aprobar a alguien que cae en impago (el caro).
    costo_fp: costo de rechazar a alguien que si habria pagado.
    """
    costo_fn = config.COSTO_FALSO_NEGATIVO if costo_fn is None else costo_fn
    costo_fp = config.COSTO_FALSO_POSITIVO if costo_fp is None else costo_fp

    _, fp, fn, _ = confusion_matrix(y_true, y_pred).ravel()
    return fn * costo_fn + fp * costo_fp


def barrer_umbrales(y_true, y_proba, costo_fn=None, costo_fp=None, pasos=101):
    """Evalua todos los umbrales de decision y calcula el costo de cada uno."""
    filas = []
    for u in np.linspace(0.01, 0.99, pasos):
        y_pred = (y_proba >= u).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        filas.append({
            "umbral": round(float(u), 4),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "falsos_negativos": int(fn),
            "falsos_positivos": int(fp),
            "costo": costo_total(y_true, y_pred, costo_fn, costo_fp),
            "tasa_rechazo": float((y_pred == 1).mean()),
        })
    return pd.DataFrame(filas)


def umbral_optimo(y_true, y_proba, costo_fn=None, costo_fp=None):
    """Umbral que minimiza el costo total, y su comparacion contra 0.5."""
    barrido = barrer_umbrales(y_true, y_proba, costo_fn, costo_fp)
    mejor = barrido.loc[barrido["costo"].idxmin()]

    y_pred_05 = (y_proba >= 0.5).astype(int)
    costo_05 = costo_total(y_true, y_pred_05, costo_fn, costo_fp)
    ahorro = costo_05 - mejor["costo"]

    return {
        "umbral": float(mejor["umbral"]),
        "costo": float(mejor["costo"]),
        "costo_umbral_05": float(costo_05),
        "ahorro_absoluto": float(ahorro),
        "ahorro_pct": float(ahorro / costo_05 * 100) if costo_05 else 0.0,
        "recall": float(mejor["recall"]),
        "precision": float(mejor["precision"]),
        "tasa_rechazo": float(mejor["tasa_rechazo"]),
        "barrido": barrido,
    }


def sensibilidad_costos(y_true, y_proba, razones=(2, 3, 5, 10, 20)):
    """Como cambia el umbral optimo segun la razon de costos supuesta.

    La razon 1:5 es un supuesto, y alguien lo va a cuestionar con razon.
    Si el umbral se mueve poco entre 1:2 y 1:20, la conclusion aguanta.
    """
    filas = []
    for r in razones:
        res = umbral_optimo(y_true, y_proba, costo_fn=float(r), costo_fp=1.0)
        filas.append({
            "razon_costos": "1:{}".format(r),
            "umbral_optimo": res["umbral"],
            "recall": round(res["recall"], 4),
            "precision": round(res["precision"], 4),
            "tasa_rechazo": round(res["tasa_rechazo"], 4),
            "ahorro_vs_05_pct": round(res["ahorro_pct"], 2),
        })
    return pd.DataFrame(filas)


def comparar_modelos(modelos, X_test, y_test):
    """Tabla comparativa de todos los modelos sobre el conjunto de prueba."""
    filas = []
    for nombre, modelo in modelos.items():
        proba = modelo.predict_proba(X_test)[:, 1]
        pred = modelo.predict(X_test)
        m = metricas(y_test, pred, proba)
        opt = umbral_optimo(y_test, proba)

        filas.append({
            "modelo": nombre,
            "auc_pr": round(m["auc_pr"], 4),
            "roc_auc": round(m["roc_auc"], 4),
            "recall": round(m["recall"], 4),
            "precision": round(m["precision"], 4),
            "f1": round(m["f1"], 4),
            "exactitud": round(m["exactitud"], 4),
            "umbral_optimo": opt["umbral"],
            "costo_optimo": round(opt["costo"], 1),
            "ahorro_vs_05_pct": round(opt["ahorro_pct"], 2),
        })

    return (pd.DataFrame(filas)
            .sort_values("auc_pr", ascending=False)
            .reset_index(drop=True))
