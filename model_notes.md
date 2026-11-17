# Notas de modelado

**Objetivo:** predecir `Delivery_Time_min` (minutos) a partir de las características del pedido.
**Análisis completo:** [`notebooks/Training.ipynb`](notebooks/Training.ipynb) (comparación de modelos) y
[`notebooks/predictive_model.ipynb`](notebooks/predictive_model.ipynb) (modelo final).
**Código de producción:** [`model_pipeline/`](model_pipeline/).

## Resumen


**Modelo elegido**: Regresión lineal (OLS) con errores estándar robustos (HC3) 
**Fórmula**: `Delivery_Time_min ~ Distance_km + Preparation_Time_min + Traffic_Level + Weather + Courier_Experience_yrs` 
**Métrica de selección**: RMSE en validación cruzada de 10 folds (`RMSE_cv`) 
**Métrica para comunicar**: MAE: *"el modelo se equivoca en promedio ±6,5 min"* 
**Test (200 pedidos)**: MAE 6,41 · RMSE 9,99 · MAPE 10,7 % · R² 0,79 
**CV con todos los datos**: MAE 6,55 · RMSE 10,59 · MAPE 10,9 % · R² 0,77 
**Modelos descartados**: GLM Gamma, GAM (normal y gamma), ElasticNet, Random Forest, Gradient Boosting 

---

## 1. Esquema de validación

```
1 000 pedidos ──► train (800) ──► selección de variables y comparación de modelos (CV de 10 folds)
              └─► test  (200) ──► evaluación final, una vez por modelo
```

- **Separación 80/20** train/test.
- **La imputación** se hace con mediana y moda
- **El modelo de producción se reentrena con los 1 000 pedidos.**.

## 2. Elección de la métrica

Se ajustaron el RMSE, MAE (es util para comunicar), MAPE (erorr relativo), AIC/BIC y R2 ajustado.
**Regla de decisión:** Para seleccionar el modelo, se tuvo en cuenta la metrica RMSE, la cual castiga los errores grandes, como prometer 30 min y tardar 60.
Si dos modelos difieren menos de `RMSE_sd` ≈ 2 min en la metrica RMSE, se consideran equivalentes y se prefiere el más simple, el cual se selecciona con el BIC.

## 3. Selección de variables (OLS)

**Método:** selección *forward*. En cada paso se agrega el término (variable o interacción) que más baja el RMSE.
- **Candidatos:** las 7 variables y 8 interacciones (distancia × tráfico, distancia × clima, distancia × preparación, tráfico × clima, entre otras).
- **Mejora mínima:** un término solo entra si baja el error de validación cruzada en al menos **0,01 min**.
- **Los cinco criterios coinciden en OLS** (RMSE_cv, MAE_cv, MAPE_cv, AIC y BIC): eligen exactamente los 5 términos.
- **Ninguna interacción mejora el modelo.**
- **`Time_of_Day` y `Vehicle_Type` quedan fuera,** como se anticipaba en la exploración.


## 4. Modelos comparados

Todos usan los mismos datos, el mismo esquema de validación y la misma imputación.

| Modelo | Selección / ajuste | RMSE_cv (train) | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|
| **OLS** | Forward con interacciones (sección 3) | **10,68 ± 1,97** | **6,41** | **9,99** | **0,790** |
| ElasticNet | Búsqueda aleatoria de α y l1_ratio, CV anidada | 10,79 ± 1,23 | 6,48 | 10,06 | 0,786 |
| GAM normal | Forward eligiendo la forma de cada variable (lineal/spline) y λ por GCV | 10,77 ± 1,20 | — | — | — |
| GAM gamma | Igual, familia gamma con enlace log | 10,89 ± 1,32 | 6,66 | 10,43 | 0,770 |
| GLM Gamma (enlace log) | Forward con interacciones, criterio devianza | 11,32 ± 1,98 | 7,05 | 10,84 | 0,752 |
| HGB (pérdida cuadrática) | Búsqueda aleatoria (25 combinaciones), CV anidada | 11,31 ± 1,11 | 7,07 | 10,63 | 0,761 |
| HGB (pérdida gamma) | Igual | 11,23 ± 1,24 | 7,11 | 10,69 | 0,759 |
| Random Forest | Búsqueda aleatoria, CV anidada | 11,59 ± 1,12 | 7,50 | 10,94 | 0,747 |

**Bootstrap pareado en test (2 000 réplicas):**

| Comparación | RMSE(modelo) − RMSE(OLS) | IC 95 % | Conclusión |
|---|---|---|---|
| GLM Gamma | +0,87 min | [0,35 ; 1,46] | OLS significativamente mejor |
| HGB gamma | +0,71 min | [0,15 ; 1,30] | OLS significativamente mejor |

### Qué dice cada comparación

- **GAM:** la selección eligió formas **lineales** para la distancia y la experiencia. Con el criterio BIC, todas las variables quedan lineales: **no hay no linealidad que aprovechar**.
- **Gradient Boosting y Random Forest:** no encuentran interacciones ni curvas que superen al OLS.
- **ElasticNet:** prácticamente empata con el OLS. La regularización no aporta, porque hay pocas variables y no son colineales.
- **GLM Gamma y modelos con enlace log:** suponen efectos **multiplicativos** (cada km suma un %). Que el OLS gane indica que los efectos son **aditivos**: cada km suma unos 3 min y el tráfico alto suma una cantidad fija de minutos.

## 5. Ajuste de hiperparámetros

OLS: La "complejidad" se ajusta con la selección de términos: Forward + RMSE_cv + mejora mínima.
GAM: Se ajusta la forma de cada variable comparando las metricas de evaluación, se fija el número de splines (10) y λ (penalización por curvatura) con **GCV** dentro de cada fold, escogiendo la del menor error fuera de muestra.
ElasticNet, Random Forest y HistGradientBoosting prueban una grilla de hiperparametros y 25 combinaciones con `RandomizedSearchCV`.
