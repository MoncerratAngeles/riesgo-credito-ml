"""Orquestador: datos -> modelos -> evaluacion -> figuras.

Ejecucion:
    python -m riesgo.pipeline
    python -m riesgo.pipeline --sin-graficas --folds 3
"""

import argparse
import json
import logging
import sys
import time

import joblib

from . import config, datos, evaluacion, modelos, viz


def configurar_log(verboso=False):
    logging.basicConfig(
        level=logging.DEBUG if verboso else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout,
    )


def ejecutar(folds=5, con_graficas=True):
    t0 = time.time()
    logging.info("=" * 64)
    logging.info("RIESGO CREDITICIO - pipeline de modelado")
    logging.info("=" * 64)

    logging.info("[1/5] Datos...")
    df = datos.preparar()
    X, y = datos.separar_xy(df)
    X_tr, X_te, y_tr, y_te = modelos.separar_train_test(X, y)

    logging.info("[2/5] Entrenamiento y validacion cruzada...")
    entrenados, cv = modelos.entrenar_todos(X_tr, y_tr, folds)

    logging.info("[3/5] Evaluacion sobre el conjunto de prueba...")
    tabla = evaluacion.comparar_modelos(entrenados, X_te, y_te)
    mejor_nombre = tabla.iloc[0]["modelo"]
    mejor = entrenados[mejor_nombre]
    proba = mejor.predict_proba(X_te)[:, 1]

    trivial = evaluacion.linea_base_trivial(y_te)
    opt = evaluacion.umbral_optimo(y_te, proba)
    sens = evaluacion.sensibilidad_costos(y_te, proba)
    imp = modelos.importancia_variables(mejor, X_te)

    logging.info("[4/5] Persistencia...")
    joblib.dump(mejor, config.MODELS / "modelo_final.joblib")
    tabla.to_csv(config.REPORTS / "comparacion_modelos.csv", index=False)
    sens.to_csv(config.REPORTS / "sensibilidad_costos.csv", index=False)
    opt["barrido"].to_csv(config.REPORTS / "barrido_umbrales.csv", index=False)
    if imp is not None:
        imp.to_csv(config.REPORTS / "importancia_variables.csv", index=False)

    resumen = {
        "mejor_modelo": mejor_nombre,
        "auc_pr": float(tabla.iloc[0]["auc_pr"]),
        "roc_auc": float(tabla.iloc[0]["roc_auc"]),
        "recall_umbral_05": float(tabla.iloc[0]["recall"]),
        "umbral_optimo": opt["umbral"],
        "ahorro_vs_05_pct": round(opt["ahorro_pct"], 2),
        "recall_umbral_optimo": round(opt["recall"], 4),
        "precision_umbral_optimo": round(opt["precision"], 4),
        "linea_base_trivial": trivial,
        "n_train": len(X_tr), "n_test": len(X_te),
        "n_features": X.shape[1],
    }
    (config.REPORTS / "resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False))

    if con_graficas:
        logging.info("[5/5] Figuras...")
        viz.distribucion_objetivo(y)
        viz.curvas_pr(entrenados, X_te, y_te)
        viz.curvas_roc(entrenados, X_te, y_te)
        viz.comparacion_modelos(tabla)
        viz.costo_vs_umbral(opt["barrido"], opt["umbral"])
        y_pred_opt = (proba >= opt["umbral"]).astype(int)
        viz.matriz_confusion(
            y_te, y_pred_opt,
            "Matriz de confusion (umbral optimo {:.2f})".format(opt["umbral"]))
        if imp is not None:
            viz.importancia(imp)
    else:
        logging.info("[5/5] Figuras omitidas.")

    _imprimir_resultados(tabla, trivial, opt, sens, imp, mejor_nombre)
    logging.info("Duracion total: %.1f s", time.time() - t0)
    return resumen


def _imprimir_resultados(tabla, trivial, opt, sens, imp, mejor):
    print("\n" + "=" * 64)
    print("COMPARACION DE MODELOS (conjunto de prueba)")
    print("=" * 64)
    print(tabla.to_string(index=False))

    print("\n" + "=" * 64)
    print("POR QUE ACCURACY NO SIRVE AQUI")
    print("=" * 64)
    print("Modelo trivial ('nadie cae en impago'):")
    print("  exactitud .................. {:.4f}".format(trivial["exactitud"]))
    print("  recall ..................... {:.4f}".format(trivial["recall"]))
    print("  impagos no detectados ...... {:,}".format(trivial["impagos_no_detectados"]))
    print("\nUn modelo inutil alcanza {:.1f}% de exactitud sin detectar".format(
        trivial["exactitud"] * 100))
    print("un solo impago. Por eso se optimiza AUC-PR y costo, no exactitud.")

    print("\n" + "=" * 64)
    print("UMBRAL OPTIMO — {}".format(mejor))
    print("=" * 64)
    print("  umbral por defecto (0.5) ... costo {:,.0f}".format(opt["costo_umbral_05"]))
    print("  umbral optimo ({:.2f}) ....... costo {:,.0f}".format(
        opt["umbral"], opt["costo"]))
    print("  ahorro ..................... {:,.0f} ({:.1f}%)".format(
        opt["ahorro_absoluto"], opt["ahorro_pct"]))
    print("  recall ..................... {:.4f}".format(opt["recall"]))
    print("  precision .................. {:.4f}".format(opt["precision"]))
    print("  tasa de rechazo ............ {:.2%}".format(opt["tasa_rechazo"]))

    print("\n" + "=" * 64)
    print("SENSIBILIDAD AL SUPUESTO DE COSTOS")
    print("=" * 64)
    print(sens.to_string(index=False))

    if imp is not None:
        print("\n" + "=" * 64)
        print("VARIABLES MAS PREDICTIVAS")
        print("=" * 64)
        print(imp.head(10).to_string(index=False))
    print()


def main():
    p = argparse.ArgumentParser(description="Modelo de riesgo crediticio.")
    p.add_argument("--folds", type=int, default=5, help="Folds de validacion cruzada")
    p.add_argument("--sin-graficas", action="store_true")
    p.add_argument("-v", "--verboso", action="store_true")
    args = p.parse_args()

    configurar_log(args.verboso)
    ejecutar(folds=args.folds, con_graficas=not args.sin_graficas)
    return 0


if __name__ == "__main__":
    sys.exit(main())
