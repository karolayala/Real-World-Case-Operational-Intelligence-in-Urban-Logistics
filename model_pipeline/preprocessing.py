"""Carga, validación, separación train/test e imputación de los datos."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def cargar_datos(ruta: Path = config.RUTA_DATOS) -> pd.DataFrame:
    """Lee el CSV y valida que tenga las columnas del modelo y la variable objetivo."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos: {ruta}. "
            "Descárgalo de Kaggle (denkuznetz/food-delivery-time-prediction) y ubícalo en data/.")
    df = pd.read_csv(ruta)
    validar_columnas(df, con_target=True)
    validar_target(df)
    return df


def validar_columnas(df: pd.DataFrame, con_target: bool = False) -> None:
    requeridas = set(config.FEATURES) | ({config.TARGET} if con_target else set())
    faltan = requeridas - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en los datos: {sorted(faltan)}")


def validar_target(df: pd.DataFrame) -> None:
    y = pd.to_numeric(df[config.TARGET], errors="coerce")
    if y.isna().any() or (y <= 0).any():
        raise ValueError(f"{config.TARGET} debe estar completa y ser mayor que 0")


def limpiar_tipos(df: pd.DataFrame, niveles: dict | None = None, avisar_en: list | None = None) -> pd.DataFrame:
    """Fuerza tipos: numéricas a float (texto -> NaN) y categóricas a object.

    Si se pasan `niveles` (categorías vistas en el entrenamiento), las categorías desconocidas se
    convierten en NaN para que se imputen con la moda. `avisar_en` limita los avisos a esas columnas.
    """
    validar_columnas(df)
    d = df.copy()
    for c in config.NUMERICAS:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    for c in config.CATEGORICAS:
        d[c] = d[c].astype(object).where(d[c].notna(), np.nan)
        if niveles is not None:
            desconocidas = ~d[c].isin(niveles[c]) & d[c].notna()
            if desconocidas.any() and (avisar_en is None or c in avisar_en):
                warnings.warn(f"{c}: valores desconocidos {sorted(d.loc[desconocidas, c].unique())} "
                              f"-> se imputan con la moda")
            d.loc[desconocidas, c] = np.nan
    return d


def dividir_train_test(df: pd.DataFrame, test_size: float = config.TEST_SIZE,
                       semilla: int = config.SEMILLA_SPLIT) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(df, test_size=test_size, random_state=semilla)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def ajustar_imputador(d: pd.DataFrame) -> dict:
    """Aprende los valores de imputación: mediana (numéricas) y moda (categóricas).

    Debe llamarse SOLO con datos de entrenamiento para evitar fuga de información.
    """
    return {**d[config.NUMERICAS].median().to_dict(),
            **d[config.CATEGORICAS].mode().iloc[0].to_dict()}


def imputar(d: pd.DataFrame, valores: dict) -> pd.DataFrame:
    return d.fillna(value=valores)
