import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from geopy.distance import geodesic
import functions as fc

# Leer el archivo CSV
df1 = pd.read_csv("../data/raw/fraudTrain.csv", index_col=0)

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

# Calcular la suma de fraudes por hora
fraudes_por_hora = df1[df1["is_fraud"] == 1].groupby("hour")["is_fraud"].sum()

# Calcular los pesos de cada hora en función de la suma de fraudes
valor_hora = fraudes_por_hora / total_fraudes

# Crear una nueva columna llamada "fraudes_por_hora" utilizando la función map()
df1["fraudes_por_hora"] = df1["hour"].map(valor_hora)

# Eliminar las columnas no deseadas del DataFrame
df1.drop(['trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last', 'street', 'city', 'zip', 'lat', 'long', 'job', 'dob', 'trans_num', 'merch_lat', 'merch_long'], inplace=True, axis=1)

# Guardar el archivo procesado
df1.to_csv("../data/processed/processed.csv")

print("Done")