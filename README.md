# Predicción del tiempo de entrega de pedidos

Modelo que estima cuántos minutos tardará un pedido en entregarse, a partir de la distancia, el tiempo de
preparación, la experiencia del repartidor, el clima y el tráfico. Incluye el análisis exploratorio, la
comparación de modelos, un pipeline de punta a punta, una API REST con aplicación web y pruebas automáticas.

**Modelo elegido:** regresión lineal (OLS) con errores robustos.

```
Delivery_Time_min ~ Distance_km + Preparation_Time_min + Traffic_Level + Weather + Courier_Experience_yrs
```

| Métrica (datos no vistos) | Test (200 pedidos) | Validación cruzada (10 folds, 1 000 pedidos) |
|---|---|---|
| MAE | 6,4 min | 6,6 min |
| RMSE | 10,0 min | 10,6 min |
| MAPE | 10,7 % | 10,9 % |
| R² | 0,79 | 0,77 |

> En promedio, el modelo se equivoca en unos **±6,5 minutos**. El error crece con la duración del pedido,
> por eso cada predicción incluye un intervalo del 90 % que se ensancha en los pedidos largos.

---

## Organización del repositorio

```
.
├── README.md                  ← este archivo
├── EDA_report.md              ← informe del análisis exploratorio
├── model_notes.md             ← informe de modelado
├── explainability.md          ← informe de explicabilidad
├── error_insights.md          ← informe de errores del modelo
├── next_steps.md              ← la API explicada y próximos pasos hacia producción
├── strategic_reflections.md   ← reflexiones estratégicas
├── requirements.txt           ← dependencias con versiones fijas (Python 3.13)
├── pytest.ini                 ← configuración de las pruebas
├── .gitignore
│
├── data/
│   └── Food_Delivery_Times.csv        ← datos de entrada (Kaggle)
│
├── model_pipeline/            ← CÓDIGO DE PRODUCCIÓN: el proceso de punta a punta
│   ├── config.py              ← rutas, variables, semillas y parámetros (un solo lugar para cambiarlos)
│   ├── preprocessing.py       ← carga, validación, limpieza de tipos, separación train/test, imputación
│   ├── model.py               ← clase ModeloTiempoEntrega (OLS + intervalos + alertas)
│   ├── train.py               ← selección forward de variables, evaluación y ajuste final
│   ├── predict.py             ← cargar el modelo y predecir (también desde la terminal)
│   └── run_pipeline.py        ← ejecuta todo el proceso con un solo comando
│
├── api/                       ← SERVICIO: API REST + aplicación web
│   ├── main.py                ← FastAPI: /predict, /predict/batch, /health, /opciones
│   └── static/index.html      ← aplicación web (un pedido en vivo, carga de CSV, gráficas)
│
├── tests/                     ← PRUEBAS AUTOMÁTICAS (40 pruebas, ~10 s)
│   ├── conftest.py            ← datos y modelo compartidos entre pruebas
│   ├── test_preprocessing.py  ← validación, limpieza, imputación sin fuga de datos
│   ├── test_model.py          ← predicciones válidas, casos especiales, persistencia, desempeño
│   ├── test_pipeline.py       ← selección reproducible, artefactos, predicción por terminal
│   └── test_api.py            ← endpoints, validación de entrada, lotes
│
├── notebooks/                 ← ANÁLISIS: la evidencia detrás de las decisiones
│   ├── Descriptivo.ipynb      ← EDA: faltantes, atípicos, análisis univariado y bivariado
│   ├── Training.ipynb         ← comparación de modelos: OLS, GLM Gamma, GAM, ElasticNet, RF, Gradient Boosting
│   ├── predictive_model.ipynb ← modelo final: p-valores, embudo de varianza, intervalos y casos especiales
│   └── Explicabilidad_y_errores.ipynb ← coeficientes, SHAP, desglose de un pedido y análisis de errores
│
├── sql/                       ← consultas SQL sobre los factores del tiempo de entrega
│   ├── sql_queries.sql
│   └── sql_insights.md
│
└── artifacts/                 ← GENERADO (no se versiona): modelo .joblib + métricas .json
```

---

## Instalación

Requiere **Python 3.13** y Git.

```bash
git clone https://github.com/karolayala/Real-World-Case-Operational-Intelligence-in-Urban-Logistics.git
cd Real-World-Case-Operational-Intelligence-in-Urban-Logistics

python -m venv env
# Windows:       env\Scripts\activate
# macOS / Linux: source env/bin/activate
pip install -r requirements.txt
```

### Inicio rápido

```bash
python -m pytest                        # 1. verifica que todo funciona (40 pruebas)
python -m model_pipeline.run_pipeline   # 2. entrena el modelo y lo guarda en artifacts/
uvicorn api.main:app --reload           # 3. levanta la API y abre http://127.0.0.1:8000/
```

## Uso

Todos los comandos se ejecutan desde la **raíz del repositorio**.

### 1. Entrenar el modelo (pipeline de punta a punta)

```bash
python -m model_pipeline.run_pipeline
```

### 2. Predecir un archivo desde la terminal

```bash
python -m model_pipeline.predict pedidos.csv predicciones.csv
```

### 3. Levantar la API y la aplicación web

```bash
uvicorn api.main:app --reload
```

| Dirección | Qué es |
|---|---|
| <http://127.0.0.1:8000/> | Aplicación web: estimación en vivo de un pedido, carga de CSV, gráficas |
| <http://127.0.0.1:8000/docs> | Documentación interactiva de la API (Swagger) |

Si `artifacts/` no existe (por ejemplo, recién clonado el repositorio), la API ejecuta el pipeline al arrancar.

| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/predict` | Un pedido: tiempo estimado, intervalo del 90 %, alertas y minutos que aporta cada factor |
| `POST` | `/predict/batch` | Lista de pedidos (hasta 10 000), cada uno con `id` opcional |
| `GET` | `/health` | Estado del servicio, fórmula y métricas del modelo |
| `GET` | `/opciones` | Categorías válidas y rangos de entrenamiento |


### 4. Ejecutar las pruebas

```bash
python -m pytest
```

### 5. Notebooks

Abre los notebooks de `notebooks/` con el entorno del proyecto. Cargan los datos con rutas relativas
(`../data/`), así que funcionan en cualquier equipo. 

## Limitaciones

- Es un prototipo: la API no tiene autenticación ni guarda las predicciones para monitorear el desempeño.

