"""Modelo de tiempo de entrega: OLS + imputación + intervalos empíricos por tramo + alertas."""
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold

from . import config
from .preprocessing import ajustar_imputador, imputar, limpiar_tipos, validar_target


def _a_python(valor):
    """Convierte tipos de numpy a tipos nativos (para serializar a JSON)."""
    if isinstance(valor, dict):
        return {k: _a_python(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_a_python(v) for v in valor]
    if isinstance(valor, np.generic):
        return valor.item()
    return valor


class ModeloTiempoEntrega:
    """Regresión lineal (OLS) para el tiempo de entrega, lista para producción.

    - Imputa faltantes con mediana/moda aprendidas en el ajuste.
    - Trata categorías desconocidas como faltantes.
    - Marca valores fuera del rango de entrenamiento y recorta predicciones < PRED_MINIMA.
    - Da intervalos de predicción empíricos por tramo de duración (más anchos en pedidos largos),
      calculados con residuos fuera de muestra.
    - Usa errores estándar robustos (HC3) para la inferencia.
    """

    def __init__(self, terminos=None, n_tramos=config.N_TRAMOS, nivel_intervalo=config.NIVEL_INTERVALO,
                 n_folds=config.N_FOLDS, semilla=config.SEMILLA_CV):
        self.terminos = list(terminos or config.TERMINOS_POR_DEFECTO)
        self.n_tramos = n_tramos
        self.nivel_intervalo = nivel_intervalo
        self.n_folds = n_folds
        self.semilla = semilla

    @property
    def formula(self) -> str:
        return f"{config.TARGET} ~ " + " + ".join(self.terminos)

    @property
    def variables_usadas(self) -> list[str]:
        """Variables que entran en la fórmula (las alertas solo se dan para estas)."""
        return [v for v in config.FEATURES if any(v in t for t in self.terminos)]

    def _tramo(self, pred):
        return np.searchsorted(self.cortes_tramos_, np.asarray(pred), side="right")

    def _preparar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia tipos, anula categorías desconocidas e imputa con los valores aprendidos."""
        d = limpiar_tipos(df, niveles=self.niveles_, avisar_en=self.variables_usadas)
        return imputar(d, self.imputacion_)

    # ---------- entrenamiento ----------
    def fit(self, df: pd.DataFrame) -> "ModeloTiempoEntrega":
        validar_target(df)
        d = limpiar_tipos(df)
        d[config.TARGET] = d[config.TARGET].astype(float)

        self.niveles_ = {c: sorted(d[c].dropna().unique()) for c in config.CATEGORICAS}
        self.rangos_ = {c: (d[c].min(), d[c].max()) for c in config.NUMERICAS}

        # 1) Predicciones fuera de muestra -> métricas e intervalos honestos
        oof = pd.Series(np.nan, index=d.index)
        kf = KFold(self.n_folds, shuffle=True, random_state=self.semilla)
        for i_tr, i_va in kf.split(d):
            tr, va = d.iloc[i_tr], d.iloc[i_va]
            vals = ajustar_imputador(tr)
            m = smf.ols(self.formula, data=imputar(tr, vals)).fit()
            oof.iloc[i_va] = m.predict(imputar(va, vals)).values
        y, err = d[config.TARGET], d[config.TARGET] - oof
        self.oof_ = pd.DataFrame({"real": y, "pred_oof": oof, "error": err})
        self.metricas_cv_ = {
            "MAE": err.abs().mean(),
            "RMSE": np.sqrt((err ** 2).mean()),
            "MAPE_%": (err.abs() / y).mean() * 100,
            "R2": 1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum(),
        }

        # 2) Intervalos empíricos por tramo de duración predicha
        _, cortes = pd.qcut(oof, self.n_tramos, retbins=True)
        self.cortes_tramos_ = cortes[1:-1]
        a = (1 - self.nivel_intervalo) / 2
        tramo = self._tramo(oof)
        self.intervalos_ = pd.DataFrame([
            {"tramo": t, "pred_desde": cortes[t], "pred_hasta": cortes[t + 1],
             "MAE": err[tramo == t].abs().mean(),
             "q_inf": err[tramo == t].quantile(a), "q_sup": err[tramo == t].quantile(1 - a)}
            for t in range(self.n_tramos)
        ]).set_index("tramo")

        # 3) Modelo final con TODOS los datos, errores robustos HC3
        self.imputacion_ = ajustar_imputador(d)
        self.modelo_ = smf.ols(self.formula, data=imputar(d, self.imputacion_)).fit(cov_type="HC3")

        self.n_entrenamiento_ = len(d)
        self.fecha_entrenamiento_ = datetime.now().isoformat(timespec="seconds")
        self.version_statsmodels_ = statsmodels.__version__
        return self

    # ---------- predicción ----------
    def predict(self, df: pd.DataFrame, intervalo: bool = True) -> pd.DataFrame:
        crudo = limpiar_tipos(df, niveles=self.niveles_, avisar_en=self.variables_usadas)
        alerta = pd.Series("", index=crudo.index)
        for c in self.variables_usadas:
            alerta[crudo[c].isna()] += f"{c} imputada; "
        for c, (lo, hi) in self.rangos_.items():
            if c in self.variables_usadas:
                fuera = (crudo[c] < lo) | (crudo[c] > hi)
                alerta[fuera] += f"{c} fuera de rango [{lo:g}, {hi:g}]; "

        pred_cruda = self.modelo_.predict(imputar(crudo, self.imputacion_))
        alerta[pred_cruda < config.PRED_MINIMA] += f"predicción recortada a {config.PRED_MINIMA:g} min; "
        pred = pred_cruda.clip(lower=config.PRED_MINIMA)

        salida = pd.DataFrame({"prediccion_min": pred}, index=crudo.index)
        if intervalo:
            t = self._tramo(pred)
            salida["limite_inf"] = (pred + self.intervalos_["q_inf"].values[t]).clip(lower=config.PRED_MINIMA)
            salida["limite_sup"] = pred + self.intervalos_["q_sup"].values[t]
        salida["alerta"] = alerta.str.rstrip("; ")
        return salida

    def contribuciones(self, fila: pd.DataFrame) -> tuple[float, list[dict]]:
        """Desglose aditivo de UNA predicción: base (intercepto) + minutos que aporta cada término."""
        x = self._preparar(fila).iloc[0]
        params = self.modelo_.params
        contribs = []
        for termino in self.terminos:
            if ":" in termino:          # interacciones: sin desglose simple
                continue
            if termino.startswith("C("):
                var = termino[2:-1]
                minutos = float(params.get(f"{termino}[T.{x[var]}]", 0.0))   # 0 = categoría de referencia
                contribs.append({"variable": var, "valor": str(x[var]), "minutos": round(minutos, 2)})
            else:
                valor = float(x[termino])
                contribs.append({"variable": termino, "valor": f"{valor:g}",
                                 "minutos": round(float(params[termino]) * valor, 2)})
        return round(float(params["Intercept"]), 2), contribs

    # ---------- información ----------
    def resumen(self) -> pd.DataFrame:
        ic = self.modelo_.conf_int()
        return pd.DataFrame({"coef": self.modelo_.params, "error_std_HC3": self.modelo_.bse,
                             "p_valor_HC3": self.modelo_.pvalues, "IC95_inf": ic[0], "IC95_sup": ic[1]})

    def ficha(self) -> dict:
        return _a_python({
            "formula": self.formula,
            "fecha_entrenamiento": self.fecha_entrenamiento_,
            "n_entrenamiento": self.n_entrenamiento_,
            "metricas_cv": {k: round(float(v), 3) for k, v in self.metricas_cv_.items()},
            "nivel_intervalo": self.nivel_intervalo,
            "intervalos_por_tramo": self.intervalos_.round(3).reset_index().to_dict(orient="records"),
            "imputacion": self.imputacion_,
            "niveles": self.niveles_,
            "rangos": self.rangos_,
            "statsmodels": self.version_statsmodels_,
        })

    # ---------- persistencia ----------
    def guardar(self, ruta: Path) -> None:
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, ruta)

    @staticmethod
    def cargar(ruta: Path) -> "ModeloTiempoEntrega":
        """Carga un modelo guardado. Solo cargar archivos generados por este pipeline (joblib usa pickle)."""
        return joblib.load(ruta)
