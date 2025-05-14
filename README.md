

🩺 Título del Proyecto:

Predicción de Gastos Médicos Usando Regresión Lineal con Ingeniería de Características

⸻

🎯 Objetivo General:

Predecir los gastos médicos de pacientes a partir de variables como edad, índice de masa corporal (IMC), tabaquismo, número de hijos y región. Para ello se utilizó un modelo de regresión lineal, complementado con técnicas de transformación y análisis de datos.

⸻

📚 Resumen de los pasos realizados:

Paso	Descripción
1	Carga de datos con Spark y conversión a Pandas.
2	Análisis exploratorio inicial y limpieza de valores extremos.
3	Visualización de relaciones entre variables con pairplot y heatmap.
4	Codificación de variables categóricas con get_dummies.
5	Entrenamiento del modelo de regresión lineal con datos estandarizados.
6	Evaluación inicial del modelo usando R² y MSE.
7	Análisis de residuales del modelo base.
8	Ingeniería de características: age², variable binaria de sobrepeso, e interacción sobrepeso*fumador.
9	Entrenamiento del modelo mejorado.
10	Evaluación del nuevo modelo con métricas y residuales.
11	Interpretación de coeficientes del modelo mejorado.
12	Informe final y conclusiones.


⸻

📈 Resultados comparativos entre modelos:

Métrica	Modelo Base	Modelo Mejorado
R²	~0.74	0.832
MSE	~0.55	0.1618
Residuales	Curvatura visible y heterocedasticidad	Más centrados, menor dispersión, sin patrón evidente

✅ El modelo mejorado superó claramente al modelo base tanto en precisión como en ajuste a los datos.

⸻

📌 Variables más influyentes (modelo mejorado):

Variable	Coeficiente	Impacto
sobrepeso*fumador	0.4940	Muy alto
smoker_yes	0.4653	Alto
age2	0.3195	Moderado-alto
children	0.0591	Bajo


⸻

🧠 Principales aprendizajes del proyecto:
	•	El análisis de residuales es clave para detectar errores sistemáticos en el modelo.
	•	La ingeniería de características (como incluir interacciones y transformaciones cuadráticas) puede mejorar significativamente un modelo simple.
	•	La regresión lineal sigue siendo una herramienta poderosa y explicativa si se prepara bien el dataset.
	•	La combinación de visualizaciones, métricas y coeficientes ofrece una visión integral y transparente del modelo.

⸻

🛠️ Posibles mejoras futuras:
	•	Aplicar modelos no lineales como regresión polinómica, Random Forest o XGBoost.
	•	Usar regularización (Ridge, Lasso) para evitar sobreajuste y mejorar la interpretabilidad.
	•	Probar con más datos o variables (por ejemplo, historial médico, nivel socioeconómico, hábitos de ejercicio).
	•	Validación cruzada para mejorar la robustez del modelo.

⸻

🎓 Conclusión Final:

Este proyecto demostró cómo pasar de un enfoque básico de predicción a uno más profundo, en el que se combinan técnicas estadísticas, visuales y conceptuales para construir un modelo mejorado, explicable y útil. Aprendiste a:
	•	Explorar y entender los datos.
	•	Modelar relaciones complejas entre variables.
	•	Evaluar y mejorar un modelo basado en evidencia.
	•	Comunicar hallazgos con claridad y solidez.

✨ Este no es solo un proyecto de datos, es una demostración de pensamiento analítico, interpretación estadística y buenas prácticas en ciencia de datos.

⸻

