"""Definicion y entrenamiento de modelos.

Cada modelo va envuelto en un Pipeline de scikit-learn junto con su
preprocesamiento. Asi el escalado y la imputacion se ajustan solo con los
datos de entrenamiento de cada fold. Si se ajustaran antes de separar
train y test, informacion del test se filtraria al modelo y las metricas
saldrian infladas.
"""

import logging

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config

log = logging.getLogger(__name__)

CATEGORICAS = ["sexo", "educacion", "estado_civil"]


def _one_hot():
    """OneHotEncoder compatible con versiones nuevas y viejas de sklearn."""
    try:
        return OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(drop="first", sparse=False, handle_unknown="ignore")


def construir_preprocesador(X):
    """Escala numericas e codifica categoricas."""
    categoricas = [c for c in CATEGORICAS if c in X.columns]
    numericas = [c for c in X.columns if c not in categoricas]

    return ColumnTransformer([
        ("num", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), numericas),
        ("cat", _one_hot(), categoricas),
    ])


def catalogo(X):
    """Los modelos a comparar, cada uno con su preprocesamiento.

    class_weight="balanced" castiga mas fuerte los errores sobre la clase
    minoritaria. Sin eso los modelos acaban prediciendo "paga" casi
    siempre, que es lo que hace el 78% de los clientes.
    """
    pre = lambda: construir_preprocesador(X)

    return {
        "regresion_logistica": Pipeline([
            ("pre", pre()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced",
                random_state=config.SEMILLA)),
        ]),
        "arbol_decision": Pipeline([
            ("pre", pre()),
            ("clf", DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=50, class_weight="balanced",
                random_state=config.SEMILLA)),
        ]),
        "random_forest": Pipeline([
            ("pre", pre()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=20,
                class_weight="balanced", n_jobs=-1,
                random_state=config.SEMILLA)),
        ]),
        "gradient_boosting": Pipeline([
            ("pre", pre()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=3,
                subsample=0.9, random_state=config.SEMILLA)),
        ]),
    }


def separar_train_test(X, y):
    """Particion estratificada: conserva la proporcion de impagos."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.SEMILLA,
    )
    log.info("Train: %d filas (%.2f%% impago) | Test: %d filas (%.2f%% impago)",
             len(X_tr), y_tr.mean() * 100, len(X_te), y_te.mean() * 100)
    return X_tr, X_te, y_tr, y_te


def validacion_cruzada(modelo, X, y, folds=5):
    """Validacion cruzada estratificada.

    La metrica principal es average_precision (area bajo la curva
    precision-recall), no accuracy ni ROC-AUC. Con 22% de positivos, la
    curva PR describe mucho mejor como le va sobre la clase que importa.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.SEMILLA)
    res = cross_validate(
        modelo, X, y, cv=cv,
        scoring=["average_precision", "roc_auc", "recall", "precision", "f1"],
        n_jobs=-1,
    )
    return {
        m: (float(np.mean(res["test_" + m])), float(np.std(res["test_" + m])))
        for m in ["average_precision", "roc_auc", "recall", "precision", "f1"]
    }


def entrenar_todos(X_tr, y_tr, folds=5):
    """Entrena y valida cada modelo del catalogo."""
    entrenados, resultados = {}, {}

    for nombre, modelo in catalogo(X_tr).items():
        log.info("Entrenando %s...", nombre)
        resultados[nombre] = validacion_cruzada(modelo, X_tr, y_tr, folds)
        modelo.fit(X_tr, y_tr)
        entrenados[nombre] = modelo
        log.info("  AUC-PR %.4f | recall %.4f | precision %.4f",
                 resultados[nombre]["average_precision"][0],
                 resultados[nombre]["recall"][0],
                 resultados[nombre]["precision"][0])

    return entrenados, resultados


def importancia_variables(modelo, X, top=15):
    """Extrae la importancia de variables del modelo entrenado."""
    import pandas as pd

    pre = modelo.named_steps["pre"]
    clf = modelo.named_steps["clf"]

    try:
        nombres = list(pre.get_feature_names_out())
    except AttributeError:
        nombres = list(X.columns)

    if hasattr(clf, "feature_importances_"):
        valores = clf.feature_importances_
        etiqueta = "importancia"
    elif hasattr(clf, "coef_"):
        valores = np.abs(clf.coef_[0])
        etiqueta = "coeficiente_abs"
    else:
        return None

    limpio = [n.split("__", 1)[-1] for n in nombres]
    return (pd.DataFrame({"variable": limpio, etiqueta: valores})
            .sort_values(etiqueta, ascending=False)
            .head(top)
            .reset_index(drop=True))
