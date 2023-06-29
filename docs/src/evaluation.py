import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score,roc_auc_score

# Cargar el modelo entrenado desde el archivo pickle

with open('../models/Model4/trained_model.pkl', 'rb') as archivo_entrada:
    model = pickle.load(archivo_entrada)

# Leer el archivo CSV de prueba
df = pd.read_csv("../data/test/test.csv")

# Obtener las características de prueba (X_test) y las etiquetas de prueba (y_test)
X_test = df[['amt', 'city_pop', 'distancia', 'fraudes_por_Categoria',
       'fraudes_por_estado', 'fraudes_por_edad', 'fraudes_por_hora',
       'fraudes_por_día']]
y_test = df['is_fraud']

# Realizar predicciones utilizando el modelo cargado
predictions = model.predict(X_test)

# Calcular la matriz de confusión
c_matrix = confusion_matrix(y_test, predictions)
print("Matriz de confusión:")
print(c_matrix)

# Mostrar la matriz de confusión como un mapa de calor
plt.figure(figsize=(10, 10))
sns.heatmap(c_matrix, annot=True)
plt.show()

# Calcular y mostrar las métricas de evaluación
print("Accuracy score:", accuracy_score(y_test, predictions))
print("Precision score:", precision_score(y_test, predictions))
print("Recall score:", recall_score(y_test, predictions))
print("ROC AUC score:", roc_auc_score(y_test, predictions))