## 1. Model Failure: Your model underestimates delivery time on rainy days. Do you fix the model, the data, or the business expectations?

Observaría por qué el modelo subestima en esos casos:
- Si es porque se tienen pocos datos de días lluviosos, lo mejor sería aumentar la cantidad de datos reales y balancear los niveles de la variable clima; si no es posile porque no hay más datos reales, haría un bootstrap con reemplazo para simular registros con dias lluviosos.
  
- Si claramente los datos explican un efecto sobre los tiempos de entrega en días lluviosos y suponiendo que hay datos suficientes para asegurar los efectos, arreglaría el modelo agregando interaccionesy utilizando algoritmos de selección de variables o efectos.
  
- Las expectativas de negocio solo las cambiaría si he probado todas las alternativas posibles. Si ha pasado eso, daría resultados en rangos para dar una idea de lo que puede suceder en días lluviosos sin fijar un solo valor estimado.

## 2. Transferability: The model performs well in Mumbai. It’s now being deployed in São Paulo. How do you ensure generalization?

- Realizo pruebas estadísticas de comparación de grupos en todas las variables entre las dos ciudades: medias, proporciones, distribuciones. Para poder asegurar un buen rendimiento en la nueva ciudad, los datos de ajuste del modelo deben tener los mismos patrones que la ciudad original.
  
- Si tengo un conjunto de datos de São Paulo, utilizo el modelo para predecir los tiempos de entrega allí y comparo las predicciones con los datos reales utilizando distintas metricas de evaluación. Así, me aseguro de que las metricas tengan un comportamiento similar a las metricas de evaluación de la ciudad original.
  
- Si tengo la opción de reentrenar el modelo con datos de São Paulo, lo hago y evaluo si ese modelo si sirve para la nueva ciudad de acuerdo a sus metricas.


## 3. GenAI Disclosure: Generative AI tools are a great resource that can facilitate development, what parts of this project did you use GenAI tools for? How did you validate or modify their output?
- Use IA para generar codigo python de graficos y algunas pruebas estadísticas, así como para crear funciones que me evitaran la repetición de código, en las fases de exploración y entrenamiento de modelos.
- Una vez seleccionado el modelo, use IA para construir la carpeta model_pipeline donde se simula su ejecución en producción. Sin embargo, todo lo relacionado con el proceso de exploración, entrenamiento y comparación de modelos tiene su notebook en la carpeta notebooks, donde realizo a fondo exploraciones.
- Use IA para generar los endpoint con FastAPI y el formulario web que brinda una mejor experiencia al usuario para el uso de estos.

Todos los resultados con IA los verifico: escojo, complemento o elimino partes del codigo según mi necesidad. Además, los resultados que me genera la IA los comparo con la historia y hallazgos que previamente haya realizado, para asegurarme de que los procesos esten bien y los resultados sean coherentes.


4. Your Signature Insight: What's one non-obvious insight or decision you're proud of from this project?

Me siento orgullosa de comparar modelos de disinto nivel de dificultad y concluir quedarme con el más sencillo: regresión lineal; no porque lo más sencillo resulte ser mejor, sino porque pude ver y entender que la forma de los datos no permitían que los modelos mejoraran: todos terminaban ajustando relaciones lineales, sin encontrar algún tipo de sesgo.

5. Going to Production: How would you deploy your model to production? What other components would you need to include/develop in your codebase? Please be as detailed each step of the process, and feel free to showcase some of these components in the codebase.
Para ir a producción, ya implementé una parte en el código:
- La logica del modelo sale de la exploración de los notebooks y pasa a la carpeta \pipeline. Allí configuro las rutas, variables y semillas base en config.py.
- Todo queda atado al comando "python -m model_pipeline.run_pipeline": En preprocessing se cargan y validan los datos, se hace la separación en train y test y se imputan los datos; en model.py guarda el modelo, los rangos de entrenamiento, los valores de imputación y las categorías validas; el train.py realiza la busqueda del mejor modelo, utilizando forward, validación cruzada, metricas de selección y entrenamiento del mejor modelo con todos los datos; el predict.py predice datos por lotes de registros.
- Generación de los artifacts en un .joblich, y las metricas en un .json
- Tener control de calidad con pruebas que eviten errores (test):  test_preprocessing.py: que la limpieza y la imputación funcionen sin fuga de datos; test_model.py: que las predicciones sean válidas y que los casos raros (faltantes, categorías desconocidas, valores extremos) se manejen bien; que el modelo se pueda guardar y cargar; que el RMSE siga por debajo de 12 min. Esta prueba falla si alguien cambia algo que empeore el modelo; test_pipeline.py: que el entrenamiento sea reproducible y genere los artefactos.
- Se desarrollo también una API con FASTAPI.
