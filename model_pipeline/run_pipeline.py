"""Pipeline de punta a punta: datos -> selección -> evaluación -> modelo final -> artefactos.

Uso (en la raíz del repositorio):
    python -m model_pipeline.run_pipeline                    # proceso completo
    python -m model_pipeline.run_pipeline --sin-seleccion    # usa los términos de config (más rápido)
    python -m model_pipeline.run_pipeline --criterio BIC
"""
import argparse
import json
import platform
import time
from pathlib import Path

import pandas as pd

from . import __version__, config
from .preprocessing import cargar_datos, dividir_train_test
from .train import entrenar_final, evaluar_en_test, evaluar_terminos, seleccion_forward


def _redondear(d: dict, n: int = 3) -> dict:
    return {k: (round(float(v), n) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
            for k, v in d.items()}


def ejecutar(ruta_datos: Path = config.RUTA_DATOS, carpeta_salida: Path = config.CARPETA_ARTEFACTOS,
             seleccionar: bool = True, criterio: str = config.CRITERIO_SELECCION, verbose: bool = True) -> dict:
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)

    log("1/5  Cargando y validando datos")
    df = cargar_datos(ruta_datos)
    log(f"     {len(df)} registros | faltantes por columna: "
        f"{ {c: int(n) for c, n in df[config.FEATURES].isna().sum().items() if n} }")

    log(f"2/5  Separando train/test ({1 - config.TEST_SIZE:.0%}/{config.TEST_SIZE:.0%}, semilla {config.SEMILLA_SPLIT})")
    train, test = dividir_train_test(df)

    if seleccionar:
        log(f"3/5  Selección forward de variables (criterio {criterio}, CV {config.N_FOLDS} folds)")
        terminos, historial = seleccion_forward(train, criterio=criterio, verbose=verbose)
    else:
        log("3/5  Selección omitida: se usan los términos de config.TERMINOS_POR_DEFECTO")
        terminos = list(config.TERMINOS_POR_DEFECTO)
        historial = pd.DataFrame([{"paso": i + 1, "agregado": t} for i, t in enumerate(terminos)])
    cv_train = evaluar_terminos(terminos, train)

    log("4/5  Evaluando en test (datos no vistos)")
    m_test = evaluar_en_test(terminos, train, test)
    log("     " + " | ".join(f"{k} = {v:.3f}" for k, v in m_test.items()))

    log("5/5  Ajustando el modelo final con todos los datos y guardando artefactos")
    modelo = entrenar_final(terminos, df)
    carpeta_salida = Path(carpeta_salida)
    ruta_modelo = config.ruta_modelo(carpeta_salida)
    modelo.guardar(ruta_modelo)

    reporte = {
        "version_modelo": config.VERSION_MODELO,
        "version_pipeline": __version__,
        "python": platform.python_version(),
        **modelo.ficha(),
        "seleccion": {
            "criterio": criterio if seleccionar else "omitida",
            "terminos": terminos,
            "pasos": historial[[c for c in ("paso", "agregado", "RMSE_cv", "MAE_cv", "BIC") if c in historial]]
                     .round(3).to_dict(orient="records"),
        },
        "metricas_cv_train": _redondear({k: v for k, v in cv_train.items() if k != "formula"}),
        "metricas_test": _redondear(m_test),
        "coeficientes_HC3": modelo.resumen().round(4).reset_index(names="termino").to_dict(orient="records"),
    }
    ruta_json = config.ruta_metricas(carpeta_salida)
    ruta_json.write_text(json.dumps(reporte, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    log(f"\nListo en {time.time() - t0:.1f} s")
    log(f"  Fórmula:          {modelo.formula}")
    log(f"  CV (todos datos): " + " | ".join(f"{k} = {v:.3f}" for k, v in modelo.metricas_cv_.items()))
    log(f"  Modelo:           {ruta_modelo}")
    log(f"  Métricas:         {ruta_json}")
    return reporte


def main(argv=None):
    p = argparse.ArgumentParser(description="Entrena el modelo de tiempo de entrega de punta a punta.")
    p.add_argument("--datos", type=Path, default=config.RUTA_DATOS, help="CSV de entrada")
    p.add_argument("--salida", type=Path, default=config.CARPETA_ARTEFACTOS, help="Carpeta de artefactos")
    p.add_argument("--sin-seleccion", action="store_true", help="Usa config.TERMINOS_POR_DEFECTO")
    p.add_argument("--criterio", default=config.CRITERIO_SELECCION,
                   choices=["RMSE_cv", "MAE_cv", "MAPE_cv", "AIC", "BIC"])
    args = p.parse_args(argv)
    ejecutar(args.datos, args.salida, seleccionar=not args.sin_seleccion, criterio=args.criterio)


if __name__ == "__main__":
    main()
