"""Figuras del analisis.

Paleta validada para daltonismo (peor separacion adyacente DeltaE 9.1 en
OKLab, umbral 8). Toda figura que use color lleva ademas leyenda o
etiqueta directa: el color nunca es el unico canal de informacion.
"""

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from . import config

log = logging.getLogger(__name__)

AZUL = "#2a78d6"
NARANJA = "#eb6834"
AQUA = "#1baf7a"
AMARILLO = "#eda100"
MAGENTA = "#e87ba4"
ROJO = "#e34948"
PALETA = [AZUL, NARANJA, AQUA, AMARILLO, MAGENTA]

INK = "#0b0b0b"
INK_SEC = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def _estilo():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_SEC,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 110,
    })


def _limpiar(ax, eje_y=True):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    if eje_y:
        ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def _guardar(fig, nombre):
    destino = config.FIGURES / "{}.png".format(nombre)
    fig.savefig(destino, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info("Figura: %s", destino.name)
    return destino


def distribucion_objetivo(y):
    """El desbalance de clases: el punto de partida del analisis."""
    _estilo()
    fig, ax = plt.subplots(figsize=(6, 4))

    conteo = y.value_counts().sort_index()
    pcts = conteo / conteo.sum() * 100
    barras = ax.bar(["Paga", "Impago"], conteo.values,
                    color=[AZUL, NARANJA], width=0.55,
                    edgecolor=SURFACE, linewidth=2)

    for b, n, p in zip(barras, conteo.values, pcts.values):
        ax.annotate("{:,}\n({:.1f}%)".format(n, p),
                    xy=(b.get_x() + b.get_width() / 2, n),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold", color=INK)

    ax.set_title("Distribucion de la variable objetivo", loc="left", color=INK)
    ax.set_ylabel("Clientes")
    ax.set_ylim(0, conteo.max() * 1.18)
    _limpiar(ax)
    fig.tight_layout()
    return _guardar(fig, "distribucion_objetivo")


def curvas_pr(modelos, X_test, y_test):
    """Curva precision-recall: la lectura correcta con clases desbalanceadas."""
    _estilo()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for (nombre, modelo), color in zip(modelos.items(), PALETA):
        proba = modelo.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, proba)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_test, proba)
        ax.plot(rec, prec, color=color, linewidth=2,
                label="{} (AUC-PR {:.3f})".format(nombre.replace("_", " "), ap))

    base = y_test.mean()
    ax.axhline(base, color=MUTED, linestyle="--", linewidth=1.2,
               label="Azar ({:.3f})".format(base))

    ax.set_xlabel("Recall (impagos detectados)")
    ax.set_ylabel("Precision (aciertos entre los marcados)")
    ax.set_title("Curva precision-recall", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SEC, loc="upper right")
    _limpiar(ax)
    fig.tight_layout()
    return _guardar(fig, "curvas_precision_recall")


def curvas_roc(modelos, X_test, y_test):
    _estilo()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for (nombre, modelo), color in zip(modelos.items(), PALETA):
        proba = modelo.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label="{} (AUC {:.3f})".format(nombre.replace("_", " "), auc))

    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.2,
            label="Azar (0.500)")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SEC, loc="lower right")
    _limpiar(ax)
    fig.tight_layout()
    return _guardar(fig, "curvas_roc")


def costo_vs_umbral(barrido, umbral_opt):
    """La figura clave: donde esta el umbral que minimiza el costo."""
    _estilo()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [1.4, 1]})

    ax1.plot(barrido["umbral"], barrido["costo"], color=AZUL, linewidth=2.2)
    fila = barrido.loc[barrido["costo"].idxmin()]
    ax1.scatter([fila["umbral"]], [fila["costo"]], s=90, color=NARANJA,
                edgecolor=SURFACE, linewidth=2, zorder=5)
    ax1.annotate("Optimo: {:.2f}\ncosto {:,.0f}".format(fila["umbral"], fila["costo"]),
                 xy=(fila["umbral"], fila["costo"]),
                 xytext=(14, 22), textcoords="offset points",
                 fontsize=10, fontweight="bold", color=INK)

    ax1.axvline(0.5, color=MUTED, linestyle="--", linewidth=1.2)
    costo_05 = float(barrido.loc[(barrido["umbral"] - 0.5).abs().idxmin(), "costo"])
    ax1.annotate("Umbral 0.5 por defecto\ncosto {:,.0f}".format(costo_05),
                 xy=(0.5, costo_05), xytext=(14, -34), textcoords="offset points",
                 fontsize=9, color=INK_SEC)

    ax1.set_ylabel("Costo total")
    ax1.set_title("Costo esperado segun el umbral de decision", loc="left", color=INK)
    _limpiar(ax1)

    ax2.plot(barrido["umbral"], barrido["recall"], color=AQUA,
             linewidth=2, label="Recall")
    ax2.plot(barrido["umbral"], barrido["precision"], color=MAGENTA,
             linewidth=2, label="Precision")
    ax2.axvline(fila["umbral"], color=NARANJA, linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Umbral de decision")
    ax2.set_ylabel("Metrica")
    ax2.legend(frameon=False, fontsize=9, labelcolor=INK_SEC)
    _limpiar(ax2)

    fig.tight_layout()
    return _guardar(fig, "costo_vs_umbral")


def matriz_confusion(y_true, y_pred, titulo="Matriz de confusion"):
    """Con etiquetas de negocio, no solo TP/FP."""
    _estilo()
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues", alpha=0.85)

    etiquetas = [
        ["Correcto\n(paga y paga)", "Falso positivo\n(rechazo innecesario)"],
        ["Falso negativo\n(impago no detectado)", "Correcto\n(impago detectado)"],
    ]
    for i in range(2):
        for j in range(2):
            fuerte = cm[i, j] > cm.max() / 2
            ax.text(j, i, "{:,}\n\n{}".format(cm[i, j], etiquetas[i][j]),
                    ha="center", va="center", fontsize=10,
                    fontweight="bold" if i == j else "normal",
                    color="#ffffff" if fuerte else INK)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predice: paga", "Predice: impago"], color=INK_SEC)
    ax.set_yticklabels(["Real: paga", "Real: impago"], color=INK_SEC)
    ax.set_title(titulo, loc="left", color=INK)
    ax.grid(False)
    for lado in ax.spines:
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    return _guardar(fig, "matriz_confusion")


def importancia(df_imp, titulo="Variables mas predictivas"):
    """Barras horizontales con valor impreso al final de cada barra."""
    _estilo()
    col = df_imp.columns[1]
    d = df_imp.sort_values(col)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(d))))
    ax.barh(d["variable"], d[col], color=AZUL, height=0.68,
            edgecolor=SURFACE, linewidth=1.5)

    for y, v in zip(range(len(d)), d[col].values):
        ax.annotate("{:.3f}".format(v), xy=(v, y), xytext=(5, 0),
                    textcoords="offset points", va="center",
                    fontsize=9, color=INK)

    ax.set_xlabel(col.replace("_", " "))
    ax.set_title(titulo, loc="left", color=INK)
    ax.set_xlim(0, d[col].max() * 1.16)
    _limpiar(ax, eje_y=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.9)
    fig.tight_layout()
    return _guardar(fig, "importancia_variables")


def comparacion_modelos(tabla):
    """Barras agrupadas: AUC-PR y recall por modelo."""
    _estilo()
    fig, ax = plt.subplots(figsize=(9, 4.5))

    x = np.arange(len(tabla))
    ancho = 0.38
    ax.bar(x - ancho / 2, tabla["auc_pr"], ancho, label="AUC-PR",
           color=AZUL, edgecolor=SURFACE, linewidth=1.5)
    ax.bar(x + ancho / 2, tabla["recall"], ancho, label="Recall",
           color=NARANJA, edgecolor=SURFACE, linewidth=1.5)

    for i, (a, r) in enumerate(zip(tabla["auc_pr"], tabla["recall"])):
        ax.annotate("{:.3f}".format(a), xy=(i - ancho / 2, a), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
        ax.annotate("{:.3f}".format(r), xy=(i + ancho / 2, r), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in tabla["modelo"]],
                       fontsize=9, color=INK_SEC)
    ax.set_ylabel("Valor")
    ax.set_ylim(0, max(tabla["auc_pr"].max(), tabla["recall"].max()) * 1.22)
    ax.set_title("Comparacion de modelos (conjunto de prueba)", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SEC)
    _limpiar(ax)
    fig.tight_layout()
    return _guardar(fig, "comparacion_modelos")
