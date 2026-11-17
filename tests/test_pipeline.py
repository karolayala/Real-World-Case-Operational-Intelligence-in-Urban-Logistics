"""Pruebas del proceso completo: selección, artefactos y predicción desde la terminal."""
import json

import pandas as pd

from model_pipeline import config
from model_pipeline.predict import main as predict_main
from model_pipeline.preprocessing import dividir_train_test
from model_pipeline.run_pipeline import ejecutar
from model_pipeline.train import seleccion_forward


def test_seleccion_reproduce_el_modelo_elegido(datos):
    train, _ = dividir_train_test(datos)
    terminos, historial = seleccion_forward(train, verbose=False)
    assert terminos == config.TERMINOS_POR_DEFECTO
    assert historial["RMSE_cv"].is_monotonic_decreasing


def test_pipeline_genera_artefactos(tmp_path):
    reporte = ejecutar(carpeta_salida=tmp_path, seleccionar=False, verbose=False)
    assert config.ruta_modelo(tmp_path).exists()
    guardado = json.loads(config.ruta_metricas(tmp_path).read_text(encoding="utf-8"))
    for clave in ("formula", "fecha_entrenamiento", "metricas_cv", "metricas_test", "coeficientes_HC3"):
        assert clave in guardado
    assert reporte["metricas_test"]["RMSE"] < config.RMSE_MAXIMO_ACEPTABLE


def test_prediccion_desde_terminal(modelo, datos, tmp_path):
    ruta_modelo = tmp_path / "m.joblib"
    modelo.guardar(ruta_modelo)
    entrada, salida = tmp_path / "entrada.csv", tmp_path / "salida.csv"
    datos.head(10)[config.FEATURES].to_csv(entrada, index=False)
    predict_main([str(entrada), str(salida), "--modelo", str(ruta_modelo)])
    res = pd.read_csv(salida)
    assert len(res) == 10
    assert {"prediccion_min", "limite_inf", "limite_sup", "alerta"} <= set(res.columns)
