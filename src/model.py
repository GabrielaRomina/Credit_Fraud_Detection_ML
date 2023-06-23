import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
import numpy as np
import pickle
from functions import *
import yaml

segmentos = []

# Leer los archivos CSV segmentados y almacenarlos en la lista
for i in range(0,4):
    segmento = pd.read_csv(f'..data/processed/segmento_{i+1}.csv')
    segmentos.append(segmento)
# Concatenar los DataFrames de los segmentos en uno solo
df1 = pd.concat(segmentos, ignore_index=True)

# Crear las variables para features y target
X = df1[['amt', 'city_pop',
       'age', 'distancia', 'day_of_week',
       'fraudes_por_Categoria', 'fraudes_por_estado', 'fraudes_por_hora']]
y = df1['is_fraud']

# Separamos una muestra del DataFrame para probar el modelo
X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size = 0.1,
                                                   random_state=0, stratify=y)

# Crear un nuevo DataFrame combinando X_test y y_test
test_data = pd.concat([X_test, y_test], axis=1)

# Guardar el archivo del test
test_data.to_csv("../data/test/test.csv")

# Aplicamos undersampling a los datos
rus = RandomUnderSampler()

X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# Separamos el nuevo conjunto en X e y
X_under = X_resampled
y_under = y_resampled

# Volvemos a separar en test y train para que el test original y los datos del undersample no contaminen a examinar
X_train_und, X_test_und, y_train_und, y_test_und = train_test_split(X_under,
                                                   y_under,
                                                   test_size = 0.2,
                                                   random_state=0)

# Crear un nuevo DataFrame combinando X_test y y_test
train_data = pd.concat([X_train_und, y_train_und], axis=1)

# Crear un nuevo DataFrame combinando X_test y y_test
train_data = pd.concat([X_train_und, y_train_und], axis=1)

segmentos = np.array_split(df1, 2)
for i, segmento in enumerate(segmentos):
    segmento.to_csv(f'..data/train/segmento_{i+1}.csv', index=False)

# Definir el pipeline
pipe = Pipeline(steps=[
    ("selectkbest", SelectKBest()),
    ("classifier", RandomForestClassifier())
])

# Definir los parámetros del GridSearchCV
rf_params = {
    'selectkbest__k': np.arange(2, 8),
    'classifier': [RandomForestClassifier()],
    'classifier__max_features': np.arange(2, 8),
    'classifier__max_depth': np.arange(2, 8),
    'classifier__class_weight': [{0: 100, 1: 1}],  # Asignar mayor peso a la clase 0
    'classifier__min_samples_leaf': np.arange(2, 8)
}

# Crear el GridSearchCV
clf = GridSearchCV(estimator=pipe, param_grid=rf_params, cv=3, scoring="roc_auc", verbose=10)

# Ajustar el GridSearchCV
clf.fit(X_train_und, y_train_und)

# Para escribir el archivo pickle
with open('../models/my_model.pkl', 'wb') as archivo_salida:
    pickle.dump(clf.best_estimator_, archivo_salida)

# Para escribir el archivo YAML
with open("../models/model_config.yaml", "w") as f:
    yaml.dump(clf.best_params_, f)