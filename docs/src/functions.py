from geopy.distance import geodesic

def calcular_distancia(row):
    """
    Calcula la distancia en kilómetros entre las coordenadas de un comercio y un cliente.

    Args:
        row (pandas.Series): Fila del DataFrame con las columnas 'lat', 'long', 'merch_lat' y 'merch_long'.

    Returns:
        float: Distancia en kilómetros.
    """
    cliente_coords = (row['lat'], row['long'])
    comercio_coords = (row['merch_lat'], row['merch_long'])
    distancia = geodesic(comercio_coords, cliente_coords).km
    return distancia

