import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from geopy.distance import geodesic
import functions as fc
import numpy as np

segmentos = []

# Leer los archivos CSV segmentados y almacenarlos en la lista
for i in range(0,4):
    segmento = pd.read_csv(f'../data/raw/segmento_{i+1}.csv')
    segmentos.append(segmento)
# Concatenar los DataFrames de los segmentos en uno solo
df1 = pd.concat(segmentos, ignore_index=True)

# Convertir la columna "dob" a tipo de dato datetime
df1["dob"] = pd.to_datetime(df1["dob"])

# Convertir la columna "trans_date_trans_time" a tipo de dato datetime
df1["trans_date_trans_time"] = pd.to_datetime(df1["trans_date_trans_time"])

# Calcular la edad a partir de la diferencia entre la fecha de transacción y la fecha de nacimiento
df1["age"] = df1.apply(lambda row: relativedelta(row["trans_date_trans_time"], row["dob"]).years, axis=1)

# Aplicar la función calcular_distancia a cada fila del DataFrame
df1['distancia'] = df1.apply(fc.calcular_distancia, axis=1)

# Generar columna con la hora (del 0 al 23) de la transacción
df1["hour"] = df1["trans_date_trans_time"].dt.hour

# Generar columna con el día de la semana (del 1 al 7) de la transacción
df1["day_of_week"] = df1["trans_date_trans_time"].dt.dayofweek + 1

# Generar columna con el día del mes (del 1 al 31) de la transacción
df1["day_of_month"] = df1["trans_date_trans_time"].dt.day

# Calcular el total de fraudes en el DataFrame
total_fraudes = df1[df1["is_fraud"] == 1]["is_fraud"].sum()

# Calcular la suma de fraudes por categoría
fraudes_por_categoria = df1[df1["is_fraud"] == 1].groupby("category")["is_fraud"].sum()

# Calcular los pesos de cada categoría en función de la suma de fraudes
valor_categoria = fraudes_por_categoria / total_fraudes

# Crear una nueva columna llamada "fraudes_por_Categoria" utilizando la función map()
df1["fraudes_por_Categoria"] = df1["category"].map(valor_categoria)

# Calcular la suma de fraudes por estado
fraudes_por_estado = df1[df1["is_fraud"] == 1].groupby("state")["is_fraud"].sum()
# Calcular los pesos de cada estado en función de la suma de fraudes
valor_estado = fraudes_por_estado / total_fraudes

# Crear una nueva columna llamada "fraudes_por_estado" utilizando la función map()
df1["fraudes_por_estado"] = df1["state"].map(valor_estado)

fraudes_por_edad = df1[df1["is_fraud"] == 1].groupby("age")["is_fraud"].sum()

# Calcular los pesos de cada hora en función de la suma de fraudes
valor_edad = fraudes_por_edad / total_fraudes

# Crear una nueva columna llamada "peso_hora" utilizando la función map()
df1["fraudes_por_edad"] = df1["age"].map(valor_edad)

# Calcular la suma de fraudes por hora
fraudes_por_hora = df1[df1["is_fraud"] == 1].groupby("hour")["is_fraud"].sum()

# Calcular los pesos de cada hora en función de la suma de fraudes
valor_hora = fraudes_por_hora / total_fraudes

# Crear una nueva columna llamada "fraudes_por_hora" utilizando la función map()
df1["fraudes_por_hora"] = df1["hour"].map(valor_hora)

fraudes_por_dia = df1[df1["is_fraud"] == 1].groupby("day_of_month")["is_fraud"].sum()

# Calcular los pesos de cada hora en función de la suma de fraudes
valor_dia = fraudes_por_dia / total_fraudes

# Crear una nueva columna llamada "peso_hora" utilizando la función map()
df1["fraudes_por_día"] = df1["day_of_month"].map(valor_dia)

# Eliminar las columnas no deseadas del DataFrame
df1.drop(['trans_date_trans_time', 'cc_num', 'merchant', 'category',
       'first', 'last', 'gender', 'street', 'city', 'state', 'zip', 'lat',
       'long', 'job', 'dob', 'trans_num', 'unix_time', 'merch_lat',
       'merch_long', 'age', 'hour', 'day_of_week', 'day_of_month'], inplace=True, axis=1)

segmentos = []

segmentos = np.array_split(df1, 3)

for i, segmento in enumerate(segmentos):
    segmento.to_csv(f'../data/processed/segmento_{i+1}.csv', index=False)

print("Done")