# Informe de análisis exploratorio (EDA)

**Datos:** `data/Food_Delivery_Times.csv` (Kaggle, *Food Delivery Time Prediction*): 1 000 pedidos y 9 columnas.
**Análisis completo:** [`notebooks/Descriptivo.ipynb`](notebooks/Descriptivo.ipynb).
---

## 1. Hallazgos principales

1. **La distancia es el factor dominante:** su correlación con el tiempo de entrega es r = 0,78 (Pearson) y ρ = 0,82 (Spearman). La relación es lineal y no muestra curvatura.
2. **Le siguen el tiempo de preparación** (r = 0,31), **el clima** (η = 0,20) y **el tráfico** (η = 0,20).
3. **La experiencia del repartidor tiene un efecto pequeño pero significativo:** r = −0,09 (p = 0,005). Más experiencia, entregas más rápidas.
4. **`Time_of_Day` y `Vehicle_Type` no se relacionan con el tiempo de entrega:** η = 0,03 en ambas.
5. **Las predictoras son prácticamente independientes entre sí:** la V de Cramér entre categóricas es ≤ 0,06. No hay riesgo de multicolinealidad.
6. **El tiempo de entrega tiene asimetría positiva:** hay pocos pedidos muy largos, que generan cola a la derecha.

---

## 2. Datos faltantes

- Las columnas Weather, Traffic_level, Time_of_Day y Courier_Experience_yrs tienen 30 datos faltantes cada una.
- **117 filas (11,7 %)** tienen al menos un faltante: 114 registros tienen uno y solo 3 registros tienen dos.
- **¿Son aleatorios?** Se compararon las filas con y sin faltante en cada una de las demás variables: **Mann-Whitney U** para las numéricas y **Chi-cuadrado** para las categóricas. Las **28 pruebas dan p ≥ 0,05**, así que no hay evidencia para rechazar la aleatoriedad. 
- **Diferencias en el tiempo medio:** cuando falta `Time_of_Day` (66,2 frente a 56,4 min) o `Traffic_Level` (62,1 frente a 56,6), la media es más alta. Sin embargo, la prueba no rechaza la aleatoriedad de los datos faltantes.

### Estrategias comparadas (regresión lineal, CV de 5 folds)

Se utilizaron 3 estrategias para determinar que hacer con los datos faltantes: Eliminar filas, imputar con la mediana y la moda e imputar con knn y la moda.
Con cada estrategia se ajustaba un modelo de regresión lineal utilizando validación cruzada para pronosticar el tiempo de entrega. 
La metrica de comparación para determinar la mejor alternativa fue el MAE.

| Estrategia | Filas usadas | MAE | Desv. estándar |
|---|---|---|---|
| Eliminar filas | 883 | 6,34 | 0,31 |
| Mediana + moda | 1 000 | 6,63 | 0,89 |
| KNN + moda | 1 000 | 6,63 | 0,88 |

**Decisión: imputar con la mediana (numéricas) y la moda (categóricas).**
- Eliminar filas da un MAE algo menor, pero se evalúa sobre un conjunto distinto (883 filas "limpias"), así que no es comparable. Además, en producción los pedidos con faltantes **seguirán llegando** y el modelo debe poder predecirlos.
- KNN no mejora a la mediana, porque las variables con faltantes casi no se relacionan con las demás. Se prefiere la opción más simple.
- La imputación se aprende **solo con los datos de entrenamiento** (dentro de cada fold en la validación cruzada) para evitar fuga de información.

---

## 3. Valores atípicos

Se identificaron 6 valores atípicos con IQR y 5 con Z-score en la variable Delivery_Time_min. 
Es de esperar que la variable Delivery_Time_min tenga atípicos, ya que el problema del proyecto es precisamente la demora en algunas entregas.

| Variable | Rango IQR [Q1 − 1,5·IQR ; Q3 + 1,5·IQR] | Atípicos (IQR) | Atípicos (\|z\| > 3) |
|---|---|---|---|
| Distance_km | [−9,8 ; 29,9] | 0 | 0 |
| Preparation_Time_min | [−7,0 ; 41,0] | 0 | 0 |
| Courier_Experience_yrs | [−5,5 ; 14,5] | 0 | 0 |
| **Delivery_Time_min** | [−4,0 ; 116,0] | **6 (0,6 %)** | 5 |

- **Los 6 atípicos están en la variable objetivo:** pedidos de 122 a 153 min (Order_ID 814, 446, 394, 385, 948, 428). Ninguno supera el límite extremo de 3·IQR (161 min).
- **Son plausibles, no errores:** corresponden a distancias largas (por ejemplo, el pedido 814 tiene 18,5 km, 29 min de preparación y 1 año de experiencia). No hay valores imposibles, como distancias o tiempos negativos.
- **Relación con los faltantes:** 2 de los 6 atípicos tienen algún faltante (`Traffic_Level` o `Time_of_Day`). Con un número tan pequeño no se puede afirmar una relación.
- **Categorías poco frecuentes:** solo `Night` está por debajo de ~10 % (8,8 %). Ninguna categoría baja del 5 %, así que no hace falta agruparlas.

**Decisión: conservar los atípicos.** Son entregas reales y largas, y eliminarlos haría que el modelo subestimara justamente los casos más costosos e importantes para el objetivo del proyecto.

---

## 4. Supuestos

- Los faltantes son aleatorios: Ninguna de las 28 pruebas lo rechaza. Observación: son solo 30 casos por columna, lo que hace que la potencia de las pruebas sea.
- Los pedidos son independientes: se utiliza validación cruzada aleatoria porque no hay dependencia temporal.
- Las relaciones de las variables con Delivery_time_min son lineales, la distancia domina las rrelaciones y no hay multicolinealidad.

