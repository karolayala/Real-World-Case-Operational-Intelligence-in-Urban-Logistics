"""Pruebas del modelo: predicciones válidas, casos especiales, persistencia, desempeño y sentido de los efectos."""
import numpy as np
import pandas as pd
import pytest

from model_pipeline import config
from model_pipeline.model import ModeloTiempoEntrega


def _pred(modelo, **campos):
    return modelo.predict(pd.DataFrame([campos])).iloc[0]


def test_una_prediccion_por_fila_sin_nan(modelo, datos):
    res = modelo.predict(datos)
    assert len(res) == len(datos)
    assert not res[["prediccion_min", "limite_inf", "limite_sup"]].isna().any().any()
    assert (res["prediccion_min"] >= config.PRED_MINIMA).all()


def test_intervalos_coherentes(modelo, datos):
    res = modelo.predict(datos)
    assert (res["limite_inf"] <= res["prediccion_min"]).all()
    assert (res["prediccion_min"] <= res["limite_sup"]).all()


def test_intervalos_mas_anchos_en_pedidos_largos(modelo):
    ancho = modelo.intervalos_["q_sup"] - modelo.intervalos_["q_inf"]
    assert ancho.iloc[-1] > ancho.iloc[0]


def test_faltantes_se_imputan_con_alerta(modelo, pedido):
    r = _pred(modelo, **{**pedido, "Weather": None, "Courier_Experience_yrs": None})
    assert "Weather imputada" in r["alerta"]
    assert "Courier_Experience_yrs imputada" in r["alerta"]


def test_categoria_nueva(modelo, pedido):
    with pytest.warns(UserWarning):
        r = _pred(modelo, **{**pedido, "Weather": "Hail"})
    assert "Weather imputada" in r["alerta"]


def test_fuera_de_rango(modelo, pedido):
    r = _pred(modelo, **{**pedido, "Distance_km": 45})
    assert "Distance_km fuera de rango" in r["alerta"]


def test_prediccion_negativa_se_recorta(modelo, pedido):
    r = _pred(modelo, **{**pedido, "Distance_km": 0, "Preparation_Time_min": 0,
                         "Courier_Experience_yrs": 40, "Traffic_Level": "Low", "Weather": "Clear"})
    assert r["prediccion_min"] == config.PRED_MINIMA
    assert "recortada" in r["alerta"]


def test_variables_no_usadas_no_generan_alertas(modelo, pedido):
    r = _pred(modelo, **{**pedido, "Time_of_Day": None, "Vehicle_Type": None})
    assert r["alerta"] == ""


def test_guardar_y_cargar_da_lo_mismo(modelo, datos, tmp_path):
    ruta = tmp_path / "modelo.joblib"
    modelo.guardar(ruta)
    cargado = ModeloTiempoEntrega.cargar(ruta)
    pd.testing.assert_frame_equal(modelo.predict(datos), cargado.predict(datos))


def test_desempeno_minimo(modelo):
    assert modelo.metricas_cv_["RMSE"] < config.RMSE_MAXIMO_ACEPTABLE
    assert modelo.metricas_cv_["R2"] > 0.7


def test_cobertura_intervalos_razonable(modelo):
    oof = modelo.oof_
    t = modelo._tramo(oof["pred_oof"])
    lo = oof["pred_oof"] + modelo.intervalos_["q_inf"].values[t]
    hi = oof["pred_oof"] + modelo.intervalos_["q_sup"].values[t]
    cobertura = ((oof["real"] >= lo) & (oof["real"] <= hi)).mean()
    assert abs(cobertura - modelo.nivel_intervalo) < 0.03


def test_sentido_de_los_efectos(modelo, pedido):
    corto = _pred(modelo, **{**pedido, "Distance_km": 3})["prediccion_min"]
    largo = _pred(modelo, **{**pedido, "Distance_km": 15})["prediccion_min"]
    bajo = _pred(modelo, **{**pedido, "Traffic_Level": "Low"})["prediccion_min"]
    alto = _pred(modelo, **{**pedido, "Traffic_Level": "High"})["prediccion_min"]
    assert largo > corto
    assert alto > bajo


def test_contribuciones_suman_la_prediccion(modelo, pedido):
    fila = pd.DataFrame([pedido])
    base, contribs = modelo.contribuciones(fila)
    total = base + sum(c["minutos"] for c in contribs)
    assert total == pytest.approx(modelo.predict(fila)["prediccion_min"].iloc[0], abs=0.05)
