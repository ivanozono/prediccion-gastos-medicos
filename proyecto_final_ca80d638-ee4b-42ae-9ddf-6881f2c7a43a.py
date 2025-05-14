# Databricks notebook source
# MAGIC %md
# MAGIC # Regresión lineal: predecir los gastos médicos de pacientes
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC Para este ejercicio utilizaremos los datos presentados en [este](https://www.kaggle.com/mirichoi0218/insurance) dataset de Kaggle en el cual se presentan datos de seguros médicos. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Descarga e instalación de librerías

# COMMAND ----------

# MAGIC %md
# MAGIC Lo primero que se hará es descargar la librería **[regressors](https://pypi.org/project/regressors/)** que ayudará a hacer un análisis más profundo sobre la regresión lineal.

# COMMAND ----------

import pandas as pd
import seaborn as sns
sns.set(style='whitegrid', context='notebook')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Descargando los datos
# MAGIC Descarguemos los datos y veamos cómo se ven.

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

df_spk = spark.read.csv("dbfs:/FileStore/tables/insurance.csv", header=True, inferSchema=True)
df_spk.show()


# COMMAND ----------

df = df_spk.toPandas()
df.head()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Analizando los datos
# MAGIC Se observará cómo se distribuyen los datos de la variable a predecir.

# COMMAND ----------

print(df.shape)
df.charges.hist(bins = 40)

# COMMAND ----------

# MAGIC %md
# MAGIC Algo que analizar, según este gráfico, es entender qué está pasando con los datos arriba de los 50,000. Parece haber muy pocos datos de este lado.

# COMMAND ----------

df[df.charges>50000]
df = df[df.charges<50000]

# COMMAND ----------

# MAGIC %md
# MAGIC En este caso, al ser pocos datos (6 de 1338), eliminaremos estos datos atípicos. A modo didáctico producen más ruido en la predicción que se está intentando hacer en este ejercicio. 
# MAGIC
# MAGIC Sin embargo es importante aclarar que **NO SE DEBEN ELIMINAR** datos atípicos sin antes conocer a alguien que conozca o sea experto en los datos para que pueda guiarnos mejor sobre ellos.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Viendo correlaciones
# MAGIC Ahora entendamos nuestros datos viendo cómo se distribuyen y correlacionan. 

# COMMAND ----------

import matplotlib.pyplot as plt
sns.pairplot(df, height=2.5)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Puntos interesantes a ver:
# MAGIC
# MAGIC - Hay 3 grupos de personas diferentes que se clasifican en edad / cargos, esto puede ser un punto a analizar después.
# MAGIC
# MAGIC En general los valores se distribuyen de manera esperada. Con valores extremos en el caso de los cargos, sin embargo esto es de esperarse pues los cargos en los hospitales pueden variar mucho por quedarse un día más en el hospital o incluso por procedimientos extras.
# MAGIC
# MAGIC - Parece que los datos están limpios, la variable de índice de masa corporal se distribuye de manera normal o gausiana, lo cual sería esperado en un índice de este tipo.

# COMMAND ----------

import numpy as np
numeric_cols = ['age', 'bmi', 'children', 'charges']
cm = np.corrcoef(df[numeric_cols].values.T)
sns.set(font_scale=1.5)
sns.heatmap(cm,annot=True, yticklabels=numeric_cols,xticklabels=numeric_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Utilizando las demás variables
# MAGIC Las demás variables son variables categoricas, sexo, fumador, región. Para poder utilizarlas utilizaremos la función **[get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)** de pandas. 
# MAGIC
# MAGIC Ahora la verás en acción

# COMMAND ----------

df = pd.get_dummies(df, columns=['sex','smoker','region'], drop_first=True)
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creando modelos 
# MAGIC Primero se usará un modelo con todas las variables.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# COMMAND ----------

X_cols = list(set(df.columns)-set(['charges']))
y_col = ['charges']

X = df[X_cols].values
y = df[y_col].values

X_train, X_test, y_train, y_test = train_test_split(X,y)
sc_x = StandardScaler().fit(X)
sc_y = StandardScaler().fit(y)

X_train = sc_x.transform(X_train)
X_test = sc_x.transform(X_test)
y_train = sc_y.transform(y_train)
y_test = sc_y.transform(y_test)

model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# COMMAND ----------

y_pred.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## Funciones de métricas
# MAGIC El siguiente punto es calcular las métricas del modelo.

# COMMAND ----------

import sklearn.metrics as metrics
mse = metrics.mean_squared_error(y_test,y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print("r2 ", r2.round(4))
print("mse: ", mse.round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC El siguiente código muestra un resumen general de los resultados.

# COMMAND ----------

# Asegurarse de que ambos arrays están aplanados (1D)
residuals = y_test.ravel() - y_pred.ravel()

# COMMAND ----------

# Ver los primeros 5 residuales
print("Primeros residuales:", residuals[:5])

# COMMAND ----------

import matplotlib.pyplot as plt

plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicciones (y_pred)")
plt.ylabel("Residuales (y_test - y_pred)")
plt.title("Gráfico de Residuales")
plt.grid(True)
plt.show()

# COMMAND ----------

print("Residuales promedio:", residuals.mean().round(4))  # Cerca de 0 en muchos casos
print("RMSE desde residuales:", np.sqrt((residuals**2).mean()).round(4))  # Igual al RMSE

# COMMAND ----------

# MAGIC %md
# MAGIC Finalmente tenemos la función que calcula los residuales. Es importante notar que es una simple resta entre los valores reales y los predichos.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Segundo modelo
# MAGIC Estos resultados de arriba son buenos, pero se pueden mejorar. Intentaremos hacer algunas transformaciones sobre las variables que pueden ser de utilidad.

# COMMAND ----------

df_second = df.copy()
df_second['age2'] = df_second.age**2
df_second['sobrepeso'] = (df_second.bmi >= 30).astype(int)
df_second['sobrepeso*fumador'] = df_second.sobrepeso * df_second.smoker_yes

# COMMAND ----------

# MAGIC %md
# MAGIC Analizando el segundo modelo

# COMMAND ----------

X_cols = ['sobrepeso*fumador', 'smoker_yes', 'age2', 'children']
y_col = ['charges']

X = df_second[X_cols].values
y = df_second[y_col].values

X_train, X_test, y_train, y_test = train_test_split(X,y)
sc_x = StandardScaler().fit(X)
sc_y = StandardScaler().fit(y)

X_train = sc_x.transform(X_train)
X_test = sc_x.transform(X_test)
y_train = sc_y.transform(y_train)
y_test = sc_y.transform(y_test)

model = LinearRegression(fit_intercept=False)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# COMMAND ----------

mse = metrics.mean_squared_error(y_test,y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print("r2 ", r2.round(4))
print("mse: ", mse.round(4))

# COMMAND ----------

residuals = y_test.ravel() - y_pred.ravel()

plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicciones (y_pred)")
plt.ylabel("Residuales")
plt.title("Gráfico de Residuales - Modelo Mejorado")
plt.grid(True)
plt.show()

# COMMAND ----------

coeficientes = model.coef_.flatten()
for col, coef in zip(X_cols, coeficientes):
    print(f"{col}: {coef:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC Interpretación variable por variable:
# MAGIC
# MAGIC ⸻
# MAGIC
# MAGIC 🔹 sobrepeso*fumador: 0.4940
# MAGIC 	•	Este coeficiente indica que un paciente que es fumador y tiene sobrepeso verá un incremento de casi medio desvío estándar en sus gastos médicos esperados.
# MAGIC 	•	Es el factor más influyente en este modelo.
# MAGIC 	•	Representa un efecto sinérgico: cuando se combinan dos factores de riesgo, el impacto es mayor que la suma de sus partes.
# MAGIC
# MAGIC 📘 Interpretación práctica:
# MAGIC Fumar y tener sobrepeso juntos elevan considerablemente los costos médicos, mucho más que cada uno por separado.
# MAGIC
# MAGIC ⸻
# MAGIC
# MAGIC 🔹 smoker_yes: 0.4653
# MAGIC 	•	Fumar por sí solo también tiene un efecto fuerte y positivo en los gastos médicos.
# MAGIC 	•	Este valor es alto, y confirma lo que se sabe clínicamente: el tabaquismo es un predictor fuerte y directo de aumento en costos médicos.
# MAGIC
# MAGIC 📘 Interpretación práctica:
# MAGIC Un paciente fumador, aún sin sobrepeso, tiene una fuerte probabilidad de incurrir en mayores costos.
# MAGIC
# MAGIC ⸻
# MAGIC
# MAGIC 🔹 age2: 0.3195
# MAGIC 	•	Este término cuadrático de la edad permite capturar una relación no lineal creciente.
# MAGIC 	•	A diferencia de un modelo con solo “edad”, este refleja que los costos no crecen de forma constante, sino que aceleran con la edad.
# MAGIC
# MAGIC 📘 Interpretación práctica:
# MAGIC A medida que envejecemos, los gastos médicos tienden a crecer más rápidamente, especialmente en edades mayores.
# MAGIC
# MAGIC ⸻
# MAGIC
# MAGIC 🔹 children: 0.0591
# MAGIC 	•	Es el coeficiente más bajo del modelo.
# MAGIC 	•	Sugeriría una relación leve y posiblemente indirecta entre la cantidad de hijos y el gasto médico.
# MAGIC
# MAGIC 📘 Interpretación práctica:
# MAGIC Tener más hijos puede asociarse a mayores necesidades médicas, pero su impacto directo es relativamente pequeño.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC 🧾  Informe Final y Conclusiones del Proyecto
# MAGIC
# MAGIC ⸻
# MAGIC
# MAGIC 🩺 Título del Proyecto:
# MAGIC
# MAGIC Predicción de Gastos Médicos Usando Regresión Lineal con Ingeniería de Características
# MAGIC
# MAGIC ⸻
# MAGIC
# MAGIC 🎯 Objetivo General:
# MAGIC
# MAGIC Predecir los gastos médicos de pacientes a partir de variables como edad, índice de masa corporal (IMC), tabaquismo, número de hijos y región. Para ello se utilizó un modelo de regresión lineal, complementado con técnicas de transformación y análisis de datos.
# MAGIC
# MAGIC ⸻
# MAGIC
# MAGIC 📚 Resumen de los pasos realizados:
# MAGIC
# MAGIC Paso	Descripción
# MAGIC 1	Carga de datos con Spark y conversión a Pandas.
# MAGIC 2	Análisis exploratorio inicial y limpieza de valores extremos.
# MAGIC 3	Visualización de relaciones entre variables con pairplot y heatmap.
# MAGIC 4	Codificación de variables categóricas con get_dummies.
# MAGIC 5	Entrenamiento del modelo de regresión lineal con datos estandarizados.
# MAGIC 6	Evaluación inicial del modelo usando R² y MSE.
# MAGIC 7	Análisis de residuales del modelo base.
# MAGIC 8	Ingeniería de características: age², variable binaria de sobrepeso, e interacción sobrepeso*fumador.
# MAGIC 9	Entrenamiento del modelo mejorado.
# MAGIC 10	Evaluación del nuevo modelo con métricas y residuales.
# MAGIC 11	Interpretación de coeficientes del modelo mejorado.
# MAGIC 12	Informe final y conclusiones.
# MAGIC
# MAGIC
# MAGIC ⸻
# MAGIC
# MAGIC 📈 Resultados comparativos entre modelos:
# MAGIC
# MAGIC Métrica	Modelo Base	Modelo Mejorado
# MAGIC R²	~0.74	0.832
# MAGIC MSE	~0.55	0.1618
# MAGIC Residuales	Curvatura visible y heterocedasticidad	Más centrados, menor dispersión, sin patrón evidente
# MAGIC
# MAGIC ✅ El modelo mejorado superó claramente al modelo base tanto en precisión como en ajuste a los datos.
# MAGIC
# MAGIC ⸻
# MAGIC
# MAGIC 📌 Variables más influyentes (modelo mejorado):
# MAGIC
# MAGIC Variable	Coeficiente	Impacto
# MAGIC sobrepeso*fumador	0.4940	Muy alto
# MAGIC smoker_yes	0.4653	Alto
# MAGIC age2	0.3195	Moderado-alto
# MAGIC children	0.0591	Bajo
# MAGIC
# MAGIC
# MAGIC ⸻
# MAGIC
# MAGIC 🧠 Principales aprendizajes del proyecto:
# MAGIC 	•	El análisis de residuales es clave para detectar errores sistemáticos en el modelo.
# MAGIC 	•	La ingeniería de características (como incluir interacciones y transformaciones cuadráticas) puede mejorar significativamente un modelo simple.
# MAGIC 	•	La regresión lineal sigue siendo una herramienta poderosa y explicativa si se prepara bien el dataset.
# MAGIC 	•	La combinación de visualizaciones, métricas y coeficientes ofrece una visión integral y transparente del modelo.
# MAGIC
# MAGIC ⸻
# MAGIC
# MAGIC 🛠️ Posibles mejoras futuras:
# MAGIC 	•	Aplicar modelos no lineales como regresión polinómica, Random Forest o XGBoost.
# MAGIC 	•	Usar regularización (Ridge, Lasso) para evitar sobreajuste y mejorar la interpretabilidad.
# MAGIC 	•	Probar con más datos o variables (por ejemplo, historial médico, nivel socioeconómico, hábitos de ejercicio).
# MAGIC 	•	Validación cruzada para mejorar la robustez del modelo.
# MAGIC
# MAGIC ⸻
# MAGIC
# MAGIC 🎓 Conclusión Final:
# MAGIC
# MAGIC Este proyecto demostró cómo pasar de un enfoque básico de predicción a uno más profundo, en el que se combinan técnicas estadísticas, visuales y conceptuales para construir un modelo mejorado, explicable y útil. Aprendiste a:
# MAGIC 	•	Explorar y entender los datos.
# MAGIC 	•	Modelar relaciones complejas entre variables.
# MAGIC 	•	Evaluar y mejorar un modelo basado en evidencia.
# MAGIC 	•	Comunicar hallazgos con claridad y solidez.
# MAGIC
# MAGIC ✨ Este no es solo un proyecto de datos, es una demostración de pensamiento analítico, interpretación estadística y buenas prácticas en ciencia de datos.
# MAGIC
# MAGIC