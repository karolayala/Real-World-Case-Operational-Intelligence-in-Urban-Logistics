# Análisis de errores: cuándo y por qué falla el modelo

**Análisis completo:** [`notebooks/Explicabilidad_y_errores.ipynb`](notebooks/Explicabilidad_y_errores.ipynb) (Parte 2).

**Cómo se midieron los errores:** cada uno de los 1 000 pedidos se predijo con un modelo que **no lo vio** al entrenar (validación cruzada de 10 folds). Así el error es el mismo que tendría el modelo con pedidos nuevos.

**Error = tiempo real − tiempo predicho:**
- **Positivo:** el pedido tardó **más** de lo predicho. El modelo se quedó corto (**subestimó**). 
- **Negativo:** el pedido llegó **antes** de lo predicho. El modelo se pasó (**sobreestimó**).

---

## Resumen

1. **El modelo no está inclinado hacia ningún lado:** su sesgo medio es prácticamente 0.
2. **Los errores no son simétricos.** Muchas veces se pasa por poco y pocas veces se queda corto por mucho. Cuando se equivoca por más de 20 min, el 91 % de las veces es porque el pedido se demoró más de lo esperado.
3. **El error crece con la distancia:** de 3,4 min en pedidos de hasta 5 km a 10,3 min en los de 15 a 20 km. Aun así, no subestima sistemáticamente ningún tipo de pedido.
4. **Los peores errores son retrasos que no se pueden anticipar con los datos disponibles:** esos pedidos se parecen al resto en todas las variables.

---

## 1. Sesgo medio: ¿el modelo tiende a quedarse corto o a pasarse?

| Indicador | Valor |
|---|---|
| Sesgo medio | **+0,02 min** (prácticamente cero) |
| Error promedio sin signo (MAE) | 6,6 min |
| Mediana del error | −1,2 min |
| Pedidos con error de ±5 min o menos | 56 % |
| Pedidos con error de ±10 min o menos | 82 % |
| Pedidos con error de ±20 min o menos | 97 % |

**Lo importante es cómo se reparte el error:**

| | % de pedidos | Error promedio |
|---|---|---|
| Se pasa (el pedido llega antes) | 59 % | 5,6 min |
| Se queda corto (el pedido tarda más) | 41 % | 8,0 min |

- **En el pedido típico, el modelo predice un poco más de lo que tarda:** la mediana del error es −1,2 min.
- **Las veces que se queda corto son menos, pero más grandes.** Por eso el promedio queda en cero: muchas sobreestimaciones pequeñas compensan pocas subestimaciones grandes.
- **El histograma del error tiene una cola larga a la derecha,** con pedidos que tardaron 40 a 60 min más de lo predicho. No hay una cola equivalente a la izquierda.
- **De los 33 pedidos con error de más de 20 min, el 91 % son subestimaciones.**

---

## 2. Error por segmento: ¿en qué pedidos se equivoca más?

Se agruparon los pedidos por cada característica y se calcularon, para cada grupo, dos medidas:
- **MAE:** cuánto se equivoca en promedio, el **tamaño** del error.
- **Sesgo:** hacia qué lado se equivoca, la **dirección** del error.

### La distancia es lo que más cambia el tamaño del error

| Distancia | Pedidos | MAE | Sesgo | 90 % de los pedidos se equivoca menos de… |
|---|---|---|---|---|
| 0–5 km | 248 | **3,4 min** | −0,2 | 5,0 min |
| 5–10 km | 241 | 5,6 min | +0,3 | 8,8 min |
| 10–15 km | 260 | 6,7 min | +0,4 | 12,7 min |
| 15–20 km | 251 | **10,3 min** | −0,5 | 17,3 min |

- **El error se multiplica por 3 entre los pedidos cortos y los largos.** En un trayecto largo hay más tramo donde puede pasar algo. Agrupando por tiempo predicho, el MAE pasa de 3,8 a 9,3 min.
- **El sesgo sigue cerca de 0 en todas las distancias.** El modelo **no subestima sistemáticamente** los pedidos largos, solo es menos preciso con ellos. 
- En las demas variables  el tamaño del error cambia poco y no hay sesgo

---

## 3. Los peores errores: ¿qué tienen en común?

### Los 10 pedidos donde más se equivocó

| Order_ID | km | Prep. (min) | Exp. (años) | Clima | Tráfico | Real | Predicho | Error |
|---|---|---|---|---|---|---|---|---|
| 394 | 15,6 | 20 | 4 | Rainy | Low | 141 | 74,5 | **+66,5** |
| 742 | 6,4 | 16 | 1 | Snowy | Low | 111 | 48,8 | **+62,2** |
| 126 | 8,8 | 12 | 3 | *faltante* | Medium | 109 | 47,0 | **+62,0** |
| 814 | 18,5 | 29 | 1 | Clear | *faltante* | 153 | 93,4 | **+59,6** |
| 45 | 5,6 | 18 | 1 | Foggy | High | 116 | 57,5 | **+58,5** |
| 446 | 19,0 | 25 | 4 | Clear | Low | 141 | 83,6 | **+57,4** |
| 813 | 2,9 | 24 | 8 | Clear | Medium | 94 | 37,6 | **+56,4** |
| 385 | 14,8 | 19 | 4 | Rainy | Low | 126 | 70,2 | **+55,8** |
| 948 | 18,0 | 10 | 7 | Clear | Medium | 123 | 69,2 | **+53,8** |
| 773 | 3,0 | 9 | 5 | Foggy | High | 90 | 37,6 | **+52,4** |

- **Todos son subestimaciones:** los 15 peores errores van de +48 a +66 min.
- **5 de los 15 peores son los atípicos identificados en el EDA** (pedidos de más de 116 min). Se conservaron porque son entregas reales, y el modelo los subestima en ~54 min.

### ¿Tienen algún perfil distinto?

Se compararon los **33 pedidos con error de más de 20 min** contra los 967 restantes, variable por variable. Un p-valor menor que 0,05 indicaría que esa variable los distingue.

| Variable | Errores grandes vs resto | p-valor |
|---|---|---|
| Distancia | 11,1 vs 10,0 km | 0,27 |
| Preparación | 15,6 vs 17,0 min | 0,26 |
| Experiencia | 3,9 vs 4,6 años | 0,20 |
| Clima | — | 0,08 |
| Tráfico | — | 0,52 |
| Franja horaria | — | 0,17 |
| Vehículo | — | 0,52 |

**Ninguna variable los distingue.** Los pedidos con errores grandes no son más lejanos, ni con peor clima, ni con repartidores nuevos: se parecen al resto.

### Diagnóstico

Los errores grandes no se deben a que el modelo esté mal planteado, sino a **retrasos que no quedan registrados en los datos**. Un modelo más complejo tampoco podría predecirlos, porque no hay ningún patrón en las variables que aprender.

---

## 4. Hallazgos

- Los errores grandes son retrasos que los datos no explican 
- El error crece con la distancia (de 3 a 10 min)
- Los pedidos con datos faltantes se subestiman (+2,6 min) 
- No hay sesgo en ningún tipo de pedido 