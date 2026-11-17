"""Prototipo de API para estimar el tiempo de entrega.

Uso (en la raíz del repositorio):
    uvicorn api.main:app --reload

- Aplicación web:          http://127.0.0.1:8000/
- Documentación (Swagger): http://127.0.0.1:8000/docs
"""
import logging
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from model_pipeline import config
from model_pipeline.predict import cargar_o_entrenar

AQUI = Path(__file__).resolve().parent
MAX_LOTE = 10_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api_entregas")


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    # Carga el modelo una sola vez al arrancar (lo entrena si no existe, p. ej. tras clonar el repo).
    # Las pruebas pueden inyectar su propio modelo en app.state.modelo antes de arrancar.
    if getattr(app.state, "modelo", None) is None:
        app.state.modelo = cargar_o_entrenar()
    log.info("Modelo cargado: %s", app.state.modelo.formula)
    yield


app = FastAPI(
    title="API de tiempo de entrega (prototipo)",
    description="Estima el tiempo de entrega de un pedido con el modelo OLS de model_pipeline.",
    version="1.0.0",
    lifespan=ciclo_de_vida,
)


# ---------- Esquemas de entrada y salida ----------
class Pedido(BaseModel):
    id: Optional[str] = Field(None, description="Identificador opcional del pedido (se devuelve tal cual)")
    Distance_km: Optional[float] = Field(None, ge=0, description="Distancia en km")
    Preparation_Time_min: Optional[float] = Field(None, ge=0, description="Tiempo de preparación (min)")
    Courier_Experience_yrs: Optional[float] = Field(None, ge=0, description="Experiencia del repartidor (años)")
    Weather: Optional[str] = Field(None, description="Clear, Foggy, Rainy, Snowy o Windy")
    Traffic_Level: Optional[str] = Field(None, description="Low, Medium o High")
    Time_of_Day: Optional[str] = Field(None, description="Morning, Afternoon, Evening o Night")
    Vehicle_Type: Optional[str] = Field(None, description="Bike, Scooter o Car")

    model_config = {"json_schema_extra": {"examples": [{
        "Distance_km": 7.5, "Preparation_Time_min": 15, "Courier_Experience_yrs": 3,
        "Weather": "Rainy", "Traffic_Level": "High", "Time_of_Day": "Evening", "Vehicle_Type": "Bike",
    }]}}


class Contribucion(BaseModel):
    variable: str
    valor: str
    minutos: float


class Prediccion(BaseModel):
    id: Optional[str] = None
    tiempo_estimado_min: float
    limite_inferior_min: float
    limite_superior_min: float
    nivel_intervalo: float
    alertas: list[str]


class PrediccionDetallada(Prediccion):
    base_min: float
    contribuciones: list[Contribucion]


# ---------- Utilidades ----------
def _predecir(modelo, pedidos: list[Pedido]):
    datos = pd.DataFrame([p.model_dump(exclude={"id"}) for p in pedidos])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return datos, modelo.predict(datos)


def _a_prediccion(modelo, pedido: Pedido, fila) -> dict:
    return {
        "id": pedido.id,
        "tiempo_estimado_min": round(float(fila["prediccion_min"]), 2),
        "limite_inferior_min": round(float(fila["limite_inf"]), 2),
        "limite_superior_min": round(float(fila["limite_sup"]), 2),
        "nivel_intervalo": modelo.nivel_intervalo,
        "alertas": [a for a in fila["alerta"].split("; ") if a],
    }


# ---------- Endpoints ----------
@app.get("/", include_in_schema=False)
def aplicacion_web():
    return FileResponse(AQUI / "static" / "index.html")


@app.get("/health")
def health(request: Request):
    ficha = request.app.state.modelo.ficha()
    return {"estado": "ok", "version_modelo": config.VERSION_MODELO, "formula": ficha["formula"],
            "fecha_entrenamiento": ficha["fecha_entrenamiento"], "metricas_cv": ficha["metricas_cv"]}


@app.get("/opciones")
def opciones(request: Request):
    """Categorías válidas y rangos de entrenamiento (los usa la aplicación web)."""
    modelo = request.app.state.modelo
    ficha = modelo.ficha()
    return {"categorias": ficha["niveles"], "rangos": ficha["rangos"], "variables_usadas": modelo.variables_usadas}


@app.post("/predict", response_model=PrediccionDetallada)
def predecir(pedido: Pedido, request: Request):
    """Un pedido: tiempo estimado, intervalo, alertas y cuántos minutos aporta cada factor."""
    modelo = request.app.state.modelo
    datos, res = _predecir(modelo, [pedido])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base, contribs = modelo.contribuciones(datos)
    resultado = PrediccionDetallada(**_a_prediccion(modelo, pedido, res.iloc[0]),
                                    base_min=base, contribuciones=contribs)
    log.info("predict | entrada=%s | salida=%.2f min", pedido.model_dump(), resultado.tiempo_estimado_min)
    return resultado


@app.post("/predict/batch", response_model=list[Prediccion])
def predecir_lote(pedidos: list[Pedido], request: Request):
    """Varios pedidos a la vez (máximo 10 000)."""
    if not pedidos:
        return []
    if len(pedidos) > MAX_LOTE:
        raise HTTPException(status_code=413, detail=f"Máximo {MAX_LOTE} pedidos por solicitud")
    modelo = request.app.state.modelo
    _, res = _predecir(modelo, pedidos)
    salida = [_a_prediccion(modelo, p, fila) for p, (_, fila) in zip(pedidos, res.iterrows())]
    log.info("predict/batch | %d pedidos", len(salida))
    return salida
