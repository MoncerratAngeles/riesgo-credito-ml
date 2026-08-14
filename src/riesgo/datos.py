"""Descarga, limpieza y preparacion del dataset de riesgo crediticio.

Tres cosas que vale la pena saber antes de leer el codigo:

1. EDUCATION 0/5/6 y MARRIAGE 0 aparecen en los datos pero el paper
   original no los define. Los agrupo en "otro" en vez de borrarlos. Son
   como 1.5% de los registros y eliminarlos meteria sesgo de seleccion.

2. La escala de atraso enganya. Los valores -2, -1 y 0 significan "sin
   consumo", "pago total" y "credito revolvente". Solo de 1 en adelante
   hay atraso real. Conservo el valor original y derivo aparte la bandera
   de atraso efectivo.

3. Aqui no se escala nada. Eso pasa dentro del Pipeline de scikit-learn,
   ajustado solo con entrenamiento, para no filtrar el test.
"""

import io
import logging
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)


def descargar(forzar=False):
    """Descarga el .xls del repositorio UCI si no existe localmente."""
    if config.ARCHIVO_CRUDO.exists() and not forzar:
        log.info("Dataset ya presente: %s", config.ARCHIVO_CRUDO.name)
        return config.ARCHIVO_CRUDO

    log.info("Descargando dataset desde UCI...")
    with urllib.request.urlopen(config.URL_DATASET, timeout=120) as resp:
        contenido = resp.read()

    with zipfile.ZipFile(io.BytesIO(contenido)) as z:
        nombre = next(n for n in z.namelist() if n.endswith(".xls"))
        config.ARCHIVO_CRUDO.write_bytes(z.read(nombre))

    log.info("Guardado: %s (%.1f MB)",
             config.ARCHIVO_CRUDO.name,
             config.ARCHIVO_CRUDO.stat().st_size / 1e6)
    return config.ARCHIVO_CRUDO


def cargar_crudo():
    """Lee el .xls. La fila 0 es un encabezado agrupado; el real es la 1."""
    ruta = descargar()
    df = pd.read_excel(ruta, header=1)
    log.info("Cargado: %d filas x %d columnas", *df.shape)
    return df


def limpiar(df):
    """Renombra, valida y normaliza categorias."""
    out = df.rename(columns=config.RENOMBRES).copy()

    if "ID" in out.columns:
        n_dupes = out["ID"].duplicated().sum()
        if n_dupes:
            log.warning("%d IDs duplicados, se eliminan.", n_dupes)
            out = out.drop_duplicates(subset="ID")
        out = out.drop(columns="ID")

    faltantes = out.isna().sum().sum()
    log.info("Valores faltantes en el dataset: %d", faltantes)

    # Categorias no documentadas -> "otro" (ver docstring del modulo).
    for col, mapa, nombre in [
        ("educacion", config.EDUCACION_MAPA, "educacion"),
        ("estado_civil", config.ESTADO_CIVIL_MAPA, "estado civil"),
        ("sexo", config.SEXO_MAPA, "sexo"),
    ]:
        antes = out[col].nunique()
        no_doc = (~out[col].isin(mapa)).sum()
        if no_doc:
            log.info("%s: %d registros con categoria no documentada -> 'otro'",
                     nombre, no_doc)
        out[col] = out[col].map(mapa).fillna("otro")
        log.info("%s: %d categorias -> %d", nombre, antes, out[col].nunique())

    # Edades imposibles: el dataset esta limpio, pero lo validamos.
    fuera = ((out["edad"] < 18) | (out["edad"] > 100)).sum()
    if fuera:
        log.warning("%d edades fuera de rango [18,100].", fuera)

    log.info("Tasa de impago: %.2f%%", out[config.OBJETIVO].mean() * 100)
    return out


def agregar_features(df):
    """Deriva variables de comportamiento de pago.

    El historial crudo mes a mes dice poco por si solo. Lo que sirve es el
    patron agregado: cuantos meses lleva atrasado, si va empeorando, que
    tanto de su limite esta usando. Es lo primero que revisaria un
    analista de riesgo a mano.
    """
    out = df.copy()

    # --- Comportamiento de atraso ---
    atrasos = out[config.MESES_ATRASO]
    out["meses_con_atraso"] = (atrasos >= 1).sum(axis=1)
    out["atraso_maximo"] = atrasos.max(axis=1)
    out["atraso_promedio"] = atrasos.mean(axis=1).round(3)
    # Tendencia: septiembre menos abril. Positivo = esta empeorando.
    out["tendencia_atraso"] = out["atraso_sep"] - out["atraso_abr"]

    # --- Utilizacion del credito ---
    # Proporcion del limite que el cliente esta usando. Es de las variables
    # mas predictivas en scoring crediticio real.
    limite = out["limite_credito"].replace(0, np.nan)
    out["utilizacion_sep"] = (out["saldo_sep"] / limite).round(4)
    out["utilizacion_promedio"] = (
        out[config.MESES_SALDO].mean(axis=1) / limite
    ).round(4)
    out["utilizacion_maxima"] = (
        out[config.MESES_SALDO].max(axis=1) / limite
    ).round(4)

    # --- Capacidad de pago ---
    # Que proporcion de lo que debe alcanza a pagar cada mes.
    total_saldo = out[config.MESES_SALDO].sum(axis=1).replace(0, np.nan)
    total_pago = out[config.MESES_PAGO].sum(axis=1)
    out["razon_pago_saldo"] = (total_pago / total_saldo).round(4)
    out["pago_promedio"] = out[config.MESES_PAGO].mean(axis=1).round(2)
    out["meses_sin_pagar"] = (out[config.MESES_PAGO] == 0).sum(axis=1)

    # --- Evolucion del saldo ---
    out["cambio_saldo"] = out["saldo_sep"] - out["saldo_abr"]
    out["saldo_promedio"] = out[config.MESES_SALDO].mean(axis=1).round(2)

    # Los infinitos vienen de divisiones por cero ya neutralizadas arriba;
    # los residuales se vuelven NaN y el imputador del pipeline los cubre.
    out = out.replace([np.inf, -np.inf], np.nan)

    nuevas = out.shape[1] - df.shape[1]
    log.info("Features derivadas: %d nuevas columnas (%d -> %d)",
             nuevas, df.shape[1], out.shape[1])
    return out


def preparar(guardar=True):
    """Ejecuta el flujo completo de datos y devuelve el DataFrame listo."""
    df = agregar_features(limpiar(cargar_crudo()))
    if guardar:
        df.to_csv(config.ARCHIVO_LIMPIO, index=False)
        log.info("Guardado: %s", config.ARCHIVO_LIMPIO.name)
    return df


def separar_xy(df):
    """Separa predictores y objetivo."""
    y = df[config.OBJETIVO]
    X = df.drop(columns=[config.OBJETIVO])
    return X, y
