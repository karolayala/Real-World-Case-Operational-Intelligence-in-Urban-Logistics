# Explicabilidad del modelo

**Modelo:** regresión lineal (OLS) con errores robustos HC3.
`Delivery_Time_min ~ Distance_km + Preparation_Time_min + Traffic_Level + Weather + Courier_Experience_yrs`
**Análisis completo:** [`notebooks/Explicabilidad_y_errores.ipynb`](notebooks/Explicabilidad_y_errores.ipynb) (Parte 1).

---

## Resumen

- **La distancia explica la mitad de lo que predice el modelo.** Le siguen el tiempo de preparación (~21 %), el tráfico y el clima (~10–12 % cada uno) y la experiencia del repartidor (~6 %).
- **Cada factor se lee directamente en minutos:** 3 min por km, casi 1 min por minuto de preparación, hasta 12 min de diferencia por el tráfico y hasta 9,5 min por el clima.

---

## 1. Coeficientes: cuántos minutos suma cada factor

Cada coeficiente es un efecto en **minutos**:
- **Variables numéricas:** cuánto cambia la entrega por cada unidad más (1 km, 1 minuto, 1 año), dejando todo lo demás igual.
- **Variables categóricas:** cuánto más o menos tarda una categoría frente a la **categoría de referencia**: tráfico `High` y clima `Clear`.

El p-valor se calcula con errores **robustos HC3**, válidos aunque el error del modelo crezca en los pedidos largos.

| Factor | Efecto | IC 95 % | p-valor |
|---|---|---|---|
| Distancia | **+2,99 min por km** | 2,87 a 3,11 | < 0,001 |
| Tiempo de preparación | **+0,97 min por minuto** | 0,88 a 1,06 | < 0,001 |
| Experiencia del repartidor | **−0,66 min por año** | −0,87 a −0,44 | < 0,001 |
| Tráfico Medium (vs High) | **−6,2 min** | −8,0 a −4,3 | < 0,001 |
| Tráfico Low (vs High) | **−11,9 min** | −13,8 a −9,9 | < 0,001 |
| Clima Snowy (vs Clear) | **+9,5 min** | 7,2 a 11,8 | < 0,001 |
| Clima Foggy (vs Clear) | **+7,3 min** | 4,8 a 9,9 | < 0,001 |
| Clima Rainy (vs Clear) | **+4,9 min** | 3,1 a 6,7 | < 0,001 |
| Clima Windy (vs Clear) | **+2,0 min** | 0,3 a 3,6 | 0,021 |
| Intercepto | 17,5 min | 14,6 a 20,3 | — |

**Lectura de negocio:**
- **Cada kilómetro cuesta unos 3 minutos,** como si el repartidor avanzara a ~20 km/h en promedio.
- **La preparación pasa casi 1 a 1 al tiempo de entrega.** El repartidor no compensa la espera en cocina, así que cada minuto que se ahorra en cocina es un minuto que gana el cliente.
- **Pasar de tráfico alto a bajo ahorra casi 12 minutos,** más que la diferencia entre nieve y cielo despejado.
- **Un repartidor con 9 años de experiencia es unos 6 minutos más rápido** que uno nuevo.

Todos los efectos son significativos y tienen el signo esperado.

---

## 2. Importancia de cada variable (SHAP)

**Qué mide SHAP:** cuánto **empuja cada variable la predicción de un pedido**, comparado con el pedido promedio (56,7 min). 
Por ejemplo, un pedido de 15 km, frente a un promedio de 10 km, queda +15 min por encima del promedio por la distancia (5 km × 3 min). 

Fórmula exacta: **coeficiente × (valor del pedido − valor promedio)**. 

| Variable | Empujón promedio | % del total | Diferencia entre un pedido "bajo" y uno "alto" (P5 a P95) |
|---|---|---|---|
| **Distancia** | ±14,8 min | **51 %** | 52,6 min |
| **Tiempo de preparación** | ±6,1 min | **21 %** | 21,3 min |
| **Tráfico** | ±3,6 min | 12 % | 11,9 min |
| **Clima** | ±3,0 min | 10 % | 9,5 min |
| **Experiencia** | ±1,6 min | 6 % | 5,9 min |

**Lectura:**
- **La distancia domina por lo mucho que varía entre pedidos,** no porque su efecto por km sea enorme. Entre un pedido cercano y uno lejano, su aporte cambia más de 50 minutos.
- **La preparación es la segunda palanca** y, a diferencia del clima o el tráfico, es **controlable** por el negocio.
- **El tráfico y el clima tienen efectos fuertes, pero solo en parte de los pedidos.** El 47 % de los pedidos son con cielo despejado y el 39 % con tráfico bajo.
- **La experiencia es significativa, pero la que menos mueve la predicción.**
- **`Time_of_Day` y `Vehicle_Type` tienen importancia cero:** la selección de variables las dejó fuera porque no mejoraban la predicción.

---

## 3. Explicación de un pedido individual

Como el modelo es una suma, **cada predicción se descompone exactamente** en minutos por factor.

**Ejemplo:** 7,5 km, 15 min de preparación, repartidor de 3 años, lluvia, tráfico alto.

| Componente | Cálculo | Minutos |
|---|---|---|
| Base (intercepto) | — | 17,5 |
| Distancia | 7,5 km × 2,99 | +22,4 |
| Preparación | 15 min × 0,97 | +14,5 |
| Tráfico | High (referencia) | 0,0 |
| Clima | Rainy | +4,9 |
| Experiencia | 3 años × −0,66 | −2,0 |
| **Tiempo estimado** | | **57,4 min** |

El intervalo del 90 % va de 43,9 a 72,3 min. 

---

## 4. Implicaciones

1. **Las palancas con más efecto son la asignación de pedidos cercanos y la reducción del tiempo de cocina.** Juntas explican ~70 % de lo que mueve la predicción.
2. **El tráfico y el clima no se controlan, pero se conocen al momento del pedido.** Sirven para ajustar la promesa al cliente: con nieve y tráfico alto, conviene prometer ~21 min más que con cielo despejado y tráfico bajo.
