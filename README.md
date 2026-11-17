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

## Informes

| Archivo | Contenido |
|---|---|
| [`EDA_report.md`](EDA_report.md) | Patrones clave, datos faltantes, atípicos y supuestos |
| [`model_notes.md`](model_notes.md) | Lógica de modelado, elección de la métrica, selección de variables y ajuste de hiperparámetros |
| [`explainability.md`](explainability.md) | Efecto de cada variable en minutos, importancia (SHAP) y cómo se explica cada predicción |
| [`error_insights.md`](error_insights.md) | Cuándo y por qué falla el modelo: sesgo, error por segmento y peores casos |

---

## Organización del repositorio

```
.
├── README.md                  ← este archivo
├── EDA_report.md              ← informe del análisis exploratorio
├── model_notes.md             ← informe de modelado
├── explainability.md          ← informe de explicabilidad
├── error_insights.md          ← informe de errores del modelo
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

**Flujo:** `data/` → `model_pipeline/` (entrena) → `artifacts/` (modelo guardado) → `api/` (lo sirve).
Los notebooks documentan cómo se llegó a cada decisión. El código de producción vive en `model_pipeline/`,
y tanto la API como el notebook `predictive_model.ipynb` importan de ahí: **una sola implementación**.

---

## Instalación

Requiere **Python 3.13**. Desde la raíz del repositorio:

```bash
python -m venv env
# Windows:       env\Scripts\activate
# macOS / Linux: source env/bin/activate
pip install -r requirements.txt
```

## Uso

Todos los comandos se ejecutan desde la **raíz del repositorio**.

### 1. Entrenar el modelo (pipeline de punta a punta)

```bash
python -m model_pipeline.run_pipeline
```

Este comando ejecuta, en unos 7 segundos:

1. **Carga y validación:** lee el CSV y verifica columnas, tipos y la variable objetivo.
2. **Separación:** divide en train y test (80/20, semilla 17).
3. **Selección de variables:** selección forward con validación cruzada de 10 folds; las interacciones son candidatas.
4. **Evaluación en test:** mide MAE, RMSE, MAPE y R² con datos no vistos.
5. **Ajuste final:** reentrena con todos los datos y guarda en `artifacts/`:
   - `modelo_tiempo_entrega_v1.joblib`: el modelo entrenado.
   - `metricas_v1.json`: fórmula, métricas, pasos de la selección, coeficientes con p-valores (HC3), valores de imputación y fecha.

Opciones:

| Opción | Efecto |
|---|---|
| `--sin-seleccion` | Usa los términos fijos de `config.py` (más rápido) |
| `--criterio BIC` | Cambia el criterio de selección (`RMSE_cv`, `MAE_cv`, `MAPE_cv`, `AIC`, `BIC`) |
| `--datos ruta.csv` | Entrena con otro archivo |
| `--salida carpeta/` | Guarda los artefactos en otra carpeta |

### 2. Predecir un archivo desde la terminal

```bash
python -m model_pipeline.predict pedidos.csv predicciones.csv
```

Agrega a cada fila `prediccion_min`, `limite_inf`, `limite_sup` (intervalo del 90 %) y `alerta`.

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

Ejemplo:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"Distance_km": 7.5, "Preparation_Time_min": 15, "Courier_Experience_yrs": 3,
       "Weather": "Rainy", "Traffic_Level": "High", "Time_of_Day": "Evening", "Vehicle_Type": "Bike"}'
```

```json
{"tiempo_estimado_min": 57.39, "limite_inferior_min": 43.86, "limite_superior_min": 72.34,
 "nivel_intervalo": 0.9, "alertas": [], "base_min": 17.49, "contribuciones": [...]}
```

### 4. Ejecutar las pruebas

```bash
python -m pytest
```

### 5. Notebooks

Abre los notebooks de `notebooks/` con el entorno del proyecto. Cargan los datos con rutas relativas
(`../data/`), así que funcionan en cualquier equipo. `Training.ipynb` tarda varios minutos por la
selección de los GAM y la validación cruzada anidada de los modelos de machine learning.

---

## Comportamiento ante datos problemáticos

| Caso | Qué hace el modelo |
|---|---|
| Campo vacío o `null` | Lo imputa con la mediana o la moda de entrenamiento y lo indica en `alerta` |
| Categoría desconocida (por ejemplo, `Weather = "Hail"`) | La trata como faltante (moda) y lo indica en `alerta` |
| Valor fuera del rango de entrenamiento (por ejemplo, 45 km) | Predice, pero marca la extrapolación en `alerta` |
| Predicción menor a 1 minuto | La recorta a 1 minuto y lo indica en `alerta` |
| Número negativo o texto en un campo numérico (API) | Error `422` con el detalle |

Las alertas solo se generan para las variables que usa el modelo: `Time_of_Day` y `Vehicle_Type` no influyen.

## Prácticas para acelerar el paso a producción

- **Una sola implementación:** la lógica del modelo está en `model_pipeline/model.py` y la usan el pipeline, la API y los notebooks.
- **Configuración centralizada** en `config.py`: rutas, variables, semillas (todas en 17), criterio de selección, umbrales.
- **Reproducible:** semillas fijas, versiones fijas en `requirements.txt` y rutas relativas.
- **Sin fuga de datos:** la imputación se aprende solo con train y dentro de cada fold.
- **Artefactos versionados:** el modelo y sus métricas llevan versión (`v1`) y fecha.
- **Pruebas automáticas:** incluyen un umbral de desempeño (RMSE < 12 min) para detectar regresiones del modelo.
- **API con validación de entrada**, documentación automática y registro de cada predicción en el log.

## Limitaciones

- Es un prototipo: la API no tiene autenticación ni guarda las predicciones para monitorear el desempeño.
- El modelo no conoce factores como la zona, el restaurante o la hora exacta, porque no están en los datos.
- `.joblib` usa *pickle*: solo cargues modelos generados por este pipeline.
