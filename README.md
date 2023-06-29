# Credit_Fraud_Detection_ML
# Modelo de clasificación para detectar fraude bancario
Este repositorio contiene un modelo de clasificación desarrollado para detectar fraudes bancarios utilizando técnicas de aprendizaje automático. El modelo se ha entrenado utilizando un conjunto de datos de transacciones bancarias etiquetadas como fraudulentas o no fraudulentas.

# Conjunto de datos
El conjunto de datos utilizado para entrenar y evaluar el modelo consiste en una colección de transacciones bancarias históricas (faker). Cada transacción está representada por una serie de atributos, como el importe, la ubicación geográfica, el tipo de transacción, entre otros. Además, cada transacción está etiquetada como "fraude" o "no fruade", lo que permite al modelo aprender a distinguir entre transacciones legítimas y fraudulentas.

# Entrenamiento del modelo
El modelo se ha entrenado utilizando una combinación de técnicas de aprendizaje automático, específicamente utilizando algoritmos de clasificación. Se han aplicado preprocesamientos y transformaciones a los datos antes de entrenar el modelo, como la normalización de los atributos numéricos y la codificación de los atributos categóricos, además de obtención de nuevas variables por procesamiento de datos.

Se han evaluado varios algoritmos de clasificación para determinar cuál ofrece el mejor rendimiento en términos de precisión y recall en la detección de fraudes. Se ha realizado una búsqueda de hiperparámetros para optimizar el rendimiento del modelo y evitar el sobreajuste.

Puesto que se partía de una base de datos no balanceada (solo un 0.6% de los datos correspondía a fraudes), se ha puesto especial énfasis en tratar el problema a partir de datos no balanceados, utilizando diferentes técnicas de undersampling y también parámetros específicos de los modelos para tratar con datos no balanceados.

# Uso del modelo
Una vez entrenado, el modelo puede utilizarse para predecir si una nueva transacción es fraudulenta o legítima. Se proporciona un script de ejemplo que carga el modelo entrenado y realiza predicciones en base a una nueva transacción. También se incluye un conjunto de datos de prueba para que puedas evaluar el rendimiento del modelo en datos no vistos previamente.

Con el archivo app.py, también se da la opción de probar a introducir nuevos datos y obtener una predicción sobre si la operación es fraudulenta o no.

**Nota**
Para ejecutar dicho archivo, verifica que tienes instaladas las librerías que contiene el documento requirements.txt.

# Fuentes
"https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
