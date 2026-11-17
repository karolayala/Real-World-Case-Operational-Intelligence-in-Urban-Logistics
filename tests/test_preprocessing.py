"""Pruebas de carga, validación, limpieza, separación e imputación."""
import numpy as np
import pandas as pd
import pytest

from model_pipeline import config
from model_pipeline.preprocessing import (ajustar_imputador, cargar_datos, dividir_train_test, imputar,
                                          limpiar_tipos, validar_columnas, validar_target)


def test_carga_datos(datos):
    assert len(datos) == 1000
    assert set(config.FEATURES + [config.TARGET]) <= set(datos.columns)


def test_archivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        cargar_datos(tmp_path / "no_existe.csv")


def test_faltan_columnas(datos):
    with pytest.raises(ValueError, match="Weather"):
        validar_columnas(datos.drop(columns=["Weather"]))


@pytest.mark.parametrize("valor", [0, -5, np.nan])
def test_target_invalido(datos, valor):
    d = datos.copy()
    d.loc[0, config.TARGET] = valor
    with pytest.raises(ValueError):
        validar_target(d)


def test_texto_en_numerica_se_vuelve_faltante(datos):
    d = datos.head(3).copy().astype({"Distance_km": object})
    d.loc[d.index[0], "Distance_km"] = "lejos"
    limpio = limpiar_tipos(d)
    assert np.isnan(limpio["Distance_km"].iloc[0])
    assert limpio["Distance_km"].dtype == float


def test_categoria_desconocida_se_vuelve_faltante(datos):
    niveles = {c: sorted(datos[c].dropna().unique()) for c in config.CATEGORICAS}
    d = datos.head(2).copy()
    d.loc[d.index[0], "Weather"] = "Hail"
    with pytest.warns(UserWarning, match="Hail"):
        limpio = limpiar_tipos(d, niveles=niveles)
    assert pd.isna(limpio["Weather"].iloc[0])
    assert limpio["Weather"].iloc[1] == d["Weather"].iloc[1]


def test_separacion_reproducible_y_sin_solape(datos):
    tr1, te1 = dividir_train_test(datos)
    tr2, te2 = dividir_train_test(datos)
    assert (len(tr1), len(te1)) == (800, 200)
    assert tr1[config.ID].tolist() == tr2[config.ID].tolist()
    assert set(tr1[config.ID]).isdisjoint(te1[config.ID])


def test_imputador_aprende_solo_de_train(datos):
    train, test = dividir_train_test(datos)
    vals = ajustar_imputador(limpiar_tipos(train))
    assert vals["Distance_km"] == train["Distance_km"].median()
    assert vals["Weather"] == train["Weather"].mode().iloc[0]
    # Cambiar test no altera lo aprendido
    test_mod = test.assign(Distance_km=999.0)
    assert ajustar_imputador(limpiar_tipos(train)) == vals
    assert imputar(limpiar_tipos(test_mod), vals)["Distance_km"].eq(999.0).all()


def test_imputacion_no_deja_faltantes(datos):
    d = limpiar_tipos(datos)
    assert d[config.FEATURES].isna().any().any()
    assert not imputar(d, ajustar_imputador(d))[config.FEATURES].isna().any().any()
