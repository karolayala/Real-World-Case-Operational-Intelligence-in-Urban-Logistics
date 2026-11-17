"""Datos y objetos compartidos por todas las pruebas (se crean una sola vez por sesión)."""
import pytest
from fastapi.testclient import TestClient

from model_pipeline.model import ModeloTiempoEntrega
from model_pipeline.preprocessing import cargar_datos


@pytest.fixture(scope="session")
def datos():
    return cargar_datos()


@pytest.fixture(scope="session")
def modelo(datos):
    """Modelo entrenado con todos los datos y los términos de config (sin tocar artifacts/)."""
    return ModeloTiempoEntrega().fit(datos)


@pytest.fixture
def pedido():
    return {"Distance_km": 7.5, "Preparation_Time_min": 15, "Courier_Experience_yrs": 3,
            "Weather": "Rainy", "Traffic_Level": "High", "Time_of_Day": "Evening", "Vehicle_Type": "Bike"}


@pytest.fixture(scope="session")
def cliente(modelo):
    """Cliente de la API con el modelo de la sesión inyectado (no lee ni escribe artifacts/)."""
    from api.main import app
    app.state.modelo = modelo
    with TestClient(app) as c:
        yield c
