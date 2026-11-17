"""Configuración central del pipeline. Cualquier cambio de rutas, variables o parámetros se hace aquí."""
from pathlib import Path

# ---------- Rutas ----------
RAIZ = Path(__file__).resolve().parents[1]
RUTA_DATOS = RAIZ / "data" / "Food_Delivery_Times.csv"
CARPETA_ARTEFACTOS = RAIZ / "artifacts"
VERSION_MODELO = "v1"


def ruta_modelo(carpeta: Path = CARPETA_ARTEFACTOS, version: str = VERSION_MODELO) -> Path:
    return Path(carpeta) / f"modelo_tiempo_entrega_{version}.joblib"


def ruta_metricas(carpeta: Path = CARPETA_ARTEFACTOS, version: str = VERSION_MODELO) -> Path:
    return Path(carpeta) / f"metricas_{version}.json"


# ---------- Variables ----------
TARGET = "Delivery_Time_min"
ID = "Order_ID"
NUMERICAS = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
CATEGORICAS = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
FEATURES = NUMERICAS + CATEGORICAS

# ---------- Separación y validación cruzada ----------
TEST_SIZE = 0.2
SEMILLA_SPLIT = 17
SEMILLA_CV = 17
N_FOLDS = 10

# ---------- Selección de variables (OLS, forward) ----------
CRITERIO_SELECCION = "RMSE_cv"          # también: "MAE_cv", "MAPE_cv", "AIC", "BIC"
# Mejora mínima para aceptar un término con criterios de validación cruzada (en sus unidades: min o %).
# Evita agregar variables por ruido: con semilla 17, C(Vehicle_Type) bajaba el RMSE_cv solo 0,003 min
# (con RMSE_sd ~ 2 min) mientras el BIC empeoraba. AIC/BIC ya penalizan la complejidad (mejora mínima 0).
MEJORA_MINIMA_CV = 0.01
TERMINOS_PRINCIPALES = [
    "Distance_km", "Preparation_Time_min", "Courier_Experience_yrs",
    "C(Weather)", "C(Traffic_Level)", "C(Time_of_Day)", "C(Vehicle_Type)",
]
# Una interacción solo puede entrar si sus efectos principales ya están (principio jerárquico)
INTERACCIONES = {
    "Distance_km:C(Traffic_Level)": ["Distance_km", "C(Traffic_Level)"],
    "Distance_km:C(Weather)": ["Distance_km", "C(Weather)"],
    "Distance_km:C(Vehicle_Type)": ["Distance_km", "C(Vehicle_Type)"],
    "Distance_km:C(Time_of_Day)": ["Distance_km", "C(Time_of_Day)"],
    "Distance_km:Preparation_Time_min": ["Distance_km", "Preparation_Time_min"],
    "Courier_Experience_yrs:C(Traffic_Level)": ["Courier_Experience_yrs", "C(Traffic_Level)"],
    "Courier_Experience_yrs:C(Vehicle_Type)": ["Courier_Experience_yrs", "C(Vehicle_Type)"],
    "C(Traffic_Level):C(Weather)": ["C(Traffic_Level)", "C(Weather)"],
}
# Términos elegidos en notebooks/Training.ipynb; se usan si se omite la selección (--sin-seleccion)
TERMINOS_POR_DEFECTO = [
    "Distance_km", "Preparation_Time_min", "C(Traffic_Level)", "C(Weather)", "Courier_Experience_yrs",
]

# ---------- Modelo final ----------
NIVEL_INTERVALO = 0.90      # intervalos de predicción empíricos
N_TRAMOS = 4                # tramos de duración para los intervalos (embudo de varianza)
PRED_MINIMA = 1.0           # ninguna predicción por debajo de 1 minuto

# ---------- Umbral de calidad (lo verifica la suite de pruebas) ----------
RMSE_MAXIMO_ACEPTABLE = 12.0
