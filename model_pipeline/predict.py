"""Predicción con el modelo guardado.

Uso desde la terminal (en la raíz del repositorio):
    python -m model_pipeline.predict entrada.csv salida.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from . import config
from .model import ModeloTiempoEntrega


def cargar_modelo(ruta: Path | None = None) -> ModeloTiempoEntrega:
    ruta = Path(ruta or config.ruta_modelo())
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el modelo {ruta}. Ejecuta primero: python -m model_pipeline.run_pipeline")
    return ModeloTiempoEntrega.cargar(ruta)


def cargar_o_entrenar(ruta: Path | None = None) -> ModeloTiempoEntrega:
    """Carga el modelo; si no existe (p. ej. recién clonado el repo), ejecuta el pipeline primero."""
    ruta = Path(ruta or config.ruta_modelo())
    if not ruta.exists():
        from .run_pipeline import ejecutar
        ejecutar(carpeta_salida=ruta.parent)
    return ModeloTiempoEntrega.cargar(ruta)


def predecir(datos: pd.DataFrame, modelo: ModeloTiempoEntrega | None = None) -> pd.DataFrame:
    """Devuelve los datos de entrada + predicción, intervalo y alertas."""
    modelo = modelo or cargar_modelo()
    return pd.concat([datos.reset_index(drop=True), modelo.predict(datos).reset_index(drop=True)], axis=1)


def main(argv=None):
    p = argparse.ArgumentParser(description="Estima el tiempo de entrega de los pedidos de un CSV.")
    p.add_argument("entrada", type=Path, help="CSV con las columnas del modelo")
    p.add_argument("salida", type=Path, help="CSV de salida con las predicciones")
    p.add_argument("--modelo", type=Path, default=None, help="Ruta del .joblib (por defecto, artifacts/)")
    args = p.parse_args(argv)

    resultado = predecir(pd.read_csv(args.entrada), cargar_modelo(args.modelo))
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(args.salida, index=False)
    print(f"{len(resultado)} predicciones guardadas en {args.salida}")


if __name__ == "__main__":
    main()
