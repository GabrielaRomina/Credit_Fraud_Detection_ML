import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest
import numpy as np
import pickle
from functions import *
import yaml
from sklearn.preprocessing import StandardScaler
import os

data_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'processed'))
segmentos = []

# Leer los archivos CSV segmentados y almacenarlos en la lista
for i in range(0,3):
    segmento = pd.read_csv(data_dir + f'\segmento_{i+1}.csv')
    segmentos.append(segmento)
# Concatenar los DataFrames de los segmentos en uno solo
df1 = pd.concat(segmentos, ignore_index=True)

# Crear las variables para features y target
X = df1[['amt', 'city_pop', 'distancia', 'fraudes_por_Categoria',
       'fraudes_por_estado', 'fraudes_por_edad', 'fraudes_por_hora',
       'fraudes_por_día']]
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

segmentos = np.array_split(df1, 2)
for i, segmento in enumerate(segmentos):
    segmento.to_csv(f'../data/train/segmento_{i+1}.csv', index=False)

# Definir el pipeline
pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("selectkbest", SelectKBest(k=6)),
    ("classifier", XGBClassifier())
])

xgb_params = {
    'classifier__learning_rate': [0.1],
    'classifier__max_depth': [10],
    'classifier__max_delta_step': [5]
}

# Crear el GridSearchCV
clf = GridSearchCV(estimator=pipe, param_grid=xgb_params, cv=3, scoring="roc_auc", verbose=10)

# Ajustar el GridSearchCV
clf.fit(X_train, y_train)

# Para escribir el archivo pickle
with open('../models/trained_model.pkl', 'wb') as archivo_salida:
    pickle.dump(clf.best_estimator_, archivo_salida)

# Para escribir el archivo YAML
with open("../models/model_config.yaml", "w") as f:
    yaml.dump(clf.best_params_, f)