"""Selección de variables (forward, con validación cruzada), evaluación en test y ajuste final."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from . import config
from .model import ModeloTiempoEntrega
from .preprocessing import ajustar_imputador, imputar, limpiar_tipos

MAYOR_ES_MEJOR = {"R2_cv", "R2_adj"}


def formula_de(terminos: list[str]) -> str:
    return f"{config.TARGET} ~ " + (" + ".join(terminos) if terminos else "1")


def metricas(y, pred) -> dict:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAPE_%": float(np.mean(np.abs((y - pred) / y)) * 100),
        "R2": r2_score(y, pred),
    }


def evaluar_terminos(terminos: list[str], datos: pd.DataFrame, n_folds: int = config.N_FOLDS,
                     semilla: int = config.SEMILLA_CV) -> dict:
    """Validación cruzada (la imputación se aprende dentro de cada fold) + métricas in-sample."""
    formula = formula_de(terminos)
    datos = limpiar_tipos(datos)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=semilla)
    folds = []
    for i_tr, i_va in kf.split(datos):
        tr, va = datos.iloc[i_tr], datos.iloc[i_va]
        vals = ajustar_imputador(tr)
        m = smf.ols(formula, data=imputar(tr, vals)).fit()
        folds.append(metricas(va[config.TARGET], m.predict(imputar(va, vals))))
    f = pd.DataFrame(folds)

    m = smf.ols(formula, data=imputar(datos, ajustar_imputador(datos))).fit()
    return {
        "formula": formula, "n_param": int(m.df_model) + 1,
        "MAE_cv": f["MAE"].mean(), "MAE_sd": f["MAE"].std(ddof=0),
        "RMSE_cv": f["RMSE"].mean(), "RMSE_sd": f["RMSE"].std(ddof=0),
        "MAPE_cv": f["MAPE_%"].mean(), "R2_cv": f["R2"].mean(),
        "R2_adj": m.rsquared_adj, "AIC": m.aic, "BIC": m.bic,
    }


def seleccion_forward(datos: pd.DataFrame, criterio: str = config.CRITERIO_SELECCION,
                      tol: float | None = None, verbose: bool = True) -> tuple[list[str], pd.DataFrame]:
    """Agrega en cada paso el término que más mejora el criterio; para cuando la mejora es menor que `tol`.

    Las interacciones solo son candidatas si sus efectos principales ya están en el modelo.
    Por defecto, `tol` = config.MEJORA_MINIMA_CV para errores de validación cruzada (en minutos o %)
    y 0 para AIC/BIC, que ya penalizan la complejidad.
    """
    if tol is None:
        tol = config.MEJORA_MINIMA_CV if criterio in ("RMSE_cv", "MAE_cv", "MAPE_cv") else 0.0
    signo = -1 if criterio in MAYOR_ES_MEJOR else 1
    seleccion = []
    actual = evaluar_terminos([], datos)
    historial = [{"paso": 0, "agregado": "(intercepto)", **actual}]

    while True:
        candidatos = [t for t in config.TERMINOS_PRINCIPALES if t not in seleccion] + \
                     [t for t, req in config.INTERACCIONES.items()
                      if t not in seleccion and all(r in seleccion for r in req)]
        if not candidatos:
            break
        resultados = [(t, evaluar_terminos(seleccion + [t], datos)) for t in candidatos]
        mejor_t, mejor = min(resultados, key=lambda x: signo * x[1][criterio])
        if signo * mejor[criterio] >= signo * actual[criterio] - tol:
            break
        seleccion.append(mejor_t)
        actual = mejor
        historial.append({"paso": len(seleccion), "agregado": mejor_t, **mejor})
        if verbose:
            print(f"   paso {len(seleccion)}: + {mejor_t:42s} {criterio} = {mejor[criterio]:.3f}")

    return seleccion, pd.DataFrame(historial)


def evaluar_en_test(terminos: list[str], train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Ajusta con train (imputación aprendida en train) y mide en test."""
    tr, te = limpiar_tipos(train), limpiar_tipos(test)
    vals = ajustar_imputador(tr)
    m = smf.ols(formula_de(terminos), data=imputar(tr, vals)).fit()
    return metricas(te[config.TARGET], m.predict(imputar(te, vals)))


def entrenar_final(terminos: list[str], df: pd.DataFrame) -> ModeloTiempoEntrega:
    """Ajusta el modelo de producción con TODOS los datos."""
    return ModeloTiempoEntrega(terminos).fit(df)
