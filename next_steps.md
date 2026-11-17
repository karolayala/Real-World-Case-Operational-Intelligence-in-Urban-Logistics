# Próximos pasos: el endpoint

Este documento explica el *ndpoint que se construyó para servir el modelo y lo que faltaría para un despliegue real en producción.

---

## Endpoint

En este proyecto, la app de pedidos le enviaría las características de un pedido a la API (un JSON), y la API respondería con el tiempo estimado de entrega.

La API está en [`api/main.py`](api/main.py) y usa **FastAPI**. Se levanta con:

```bash
uvicorn api.main:app --reload
```

| Dirección | Qué es |
|---|---|
| <http://127.0.0.1:8000/> | Aplicación web: formulario de un pedido, carga de CSV y gráficas |
| <http://127.0.0.1:8000/docs> | Documentación interactiva (Swagger): permite probar cada endpoint desde el navegador |

### Cómo funciona por dentro

1. **Al arrancar,** la API carga el modelo entrenado (`artifacts/modelo_tiempo_entrega_v1.joblib`) **una sola vez** y lo deja en memoria. Si el archivo no existe, por ejemplo recién clonado el repositorio, ejecuta el pipeline de entrenamiento automáticamente.
2. **Al recibir un pedido,** FastAPI valida los datos con un esquema (`Pedido`): tipos correctos y números no negativos.
3. **El modelo predice:** imputa los faltantes, revisa las categorías y los rangos, calcula el tiempo y el intervalo del 90 %.
4. **La API responde** en formato JSON y registra la entrada y la salida en el log.

---

## 2. Los endpoints

### `POST /predict`: un pedido

Es el endpoint principal. Recibe un pedido y devuelve el tiempo estimado, el intervalo, las alertas y **cuántos minutos aporta cada factor**.

**Entrada.** Todos los campos son opcionales; si falta alguno, se imputa:

```json
{
  "Distance_km": 7.5,
  "Preparation_Time_min": 15,
  "Courier_Experience_yrs": 3,
  "Weather": "Rainy",
  "Traffic_Level": "High",
  "Time_of_Day": "Evening",
  "Vehicle_Type": "Bike"
}
```

**Respuesta:**

```json
{
  "id": null,
  "tiempo_estimado_min": 57.39,
  "limite_inferior_min": 43.86,
  "limite_superior_min": 72.34,
  "nivel_intervalo": 0.9,
  "alertas": [],
  "base_min": 17.49,
  "contribuciones": [
    {"variable": "Distance_km",            "valor": "7.5",   "minutos": 22.43},
    {"variable": "Preparation_Time_min",   "valor": "15",    "minutos": 14.52},
    {"variable": "Traffic_Level",          "valor": "High",  "minutos": 0.0},
    {"variable": "Weather",                "valor": "Rainy", "minutos": 4.92},
    {"variable": "Courier_Experience_yrs", "valor": "3",     "minutos": -1.97}
  ]
}
```

### `POST /predict/batch`: varios pedidos

Recibe una lista de pedidos, hasta 10 000, cada uno con un `id` opcional que se devuelve tal cual para poder cruzar los resultados. No incluye el desglose por factor, para que la respuesta sea liviana.

**Entrada:**

```json
[
  {"id": "A1", "Distance_km": 3,  "Preparation_Time_min": 10, "Courier_Experience_yrs": 5,
   "Weather": "Clear", "Traffic_Level": "Low"},
  {"id": "A2", "Distance_km": 25, "Preparation_Time_min": 20,
   "Weather": "Hail",  "Traffic_Level": "High"}
]
```

**Respuesta:**

```json
[
  {"id": "A1", "tiempo_estimado_min": 21.01, "limite_inferior_min": 15.66, "limite_superior_min": 27.88,
   "nivel_intervalo": 0.9, "alertas": []},
  {"id": "A2", "tiempo_estimado_min": 108.34, "limite_inferior_min": 91.11, "limite_superior_min": 123.78,
   "nivel_intervalo": 0.9,
   "alertas": ["Courier_Experience_yrs imputada", "Weather imputada", "Distance_km fuera de rango [0.59, 19.99]"]}
]
```

El pedido `A2` muestra cómo se manejan los datos problemáticos. En lugar de fallar, el modelo predice y **avisa** qué pasó:
- **Falta la experiencia:** se imputa con la mediana.
- **`"Hail"` (granizo) no es una categoría conocida:** se trata como faltante y se imputa con la moda.
- **25 km está fuera de lo visto en entrenamiento (0,6–20 km):** la predicción es una extrapolación y conviene tomarla con cautela.

### `GET /health`: estado del servicio

Responde si la API está funcionando y qué modelo tiene cargado. Los servidores lo consultan periódicamente para saber si deben reiniciar el servicio.

```json
{
  "estado": "ok",
  "version_modelo": "v1",
  "formula": "Delivery_Time_min ~ Distance_km + Preparation_Time_min + C(Traffic_Level) + C(Weather) + Courier_Experience_yrs",
  "fecha_entrenamiento": "2026-11-17T12:08:10",
  "metricas_cv": {"MAE": 6.552, "RMSE": 10.594, "MAPE_%": 10.911, "R2": 0.769}
}
```

### `GET /opciones`: valores válidos

Devuelve las categorías aceptadas, los rangos de entrenamiento y las variables que usa el modelo. La aplicación web lo usa para armar el formulario, y sirve a cualquier sistema que necesite saber qué enviar.

### Errores

| Caso | Respuesta |
|---|---|
| Número negativo o texto en un campo numérico | `422`, con el detalle del campo, por ejemplo `"Distance_km": "Input should be greater than or equal to 0"` |
| Más de 10 000 pedidos en `/predict/batch` | `413`, con el mensaje `"Máximo 10000 pedidos por solicitud"` |
| Campo vacío, categoría desconocida o valor fuera de rango | `200`: predice y lo indica en `alertas` |

**Criterio de diseño:** solo se rechazan los datos **imposibles**, como una distancia negativa. Los datos **incompletos o inusuales** se predicen con aviso, porque en la operación real los pedidos siguen llegando aunque falte un dato.

### Ejemplos de uso

Desde la terminal:

Desde Python:

```python
import requests
r = requests.post("http://127.0.0.1:8000/predict",
                  json={"Distance_km": 7.5, "Preparation_Time_min": 15, "Weather": "Rainy"})
print(r.json()["tiempo_estimado_min"])
```

### Pruebas del endpoint

[`tests/test_api.py`](tests/test_api.py) tiene 13 pruebas automáticas de la API:
- cada endpoint responde con el formato esperado;
- las contribuciones suman la predicción;
- las entradas inválidas se rechazan con 422;
- los faltantes y las categorías desconocidas generan alertas;
- los lotes devuelven el `id` y dan el mismo resultado que los pedidos individuales;
- un lote vacío devuelve una lista vacía.

```bash
python -m pytest tests/test_api.py
```

---

## 3. Lo que faltaría para producción

La API actal es un **prototipo funcional**: corre, valida, predice y explica . Para un despliegue real faltarían:

- Empaquetar el servicio (Docker): Un `Dockerfile` que instale las dependencias, copie el código y el modelo, y levante la API. Así el servicio corre igual en cualquier servidor o en la nube (AWS, GCP o Azure), sin depender de la configuración de un equipo.
- Automatizar cambios en producción, una vez se hagan cambios en el repositorio y se pasen pruebas de validación.
- Monitorear el modelo, guardando las predicciones y el tiempo real de entrega una vez se tenga y validar las metricas de validación. Hace falta tener la posibilidad de reentrenar si cambian los patrones de los datos de entrada.
- Garantizar la seguridad con metodos de autenticación

