"""Pipeline de punta a punta para predecir el tiempo de entrega de pedidos.

Módulos:
    config         Rutas, variables, semillas y parámetros (un solo lugar para cambiarlos).
    preprocessing  Carga, validación de columnas/tipos, separación train/test e imputación.
    model          Clase ModeloTiempoEntrega (OLS + intervalos + alertas).
    train          Selección de variables, evaluación y ajuste final.
    predict        Cargar el modelo guardado y predecir (también desde la terminal).
    run_pipeline   Ejecuta todo el proceso con un solo comando.
"""
__version__ = "1.0.0"
