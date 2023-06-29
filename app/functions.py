from geopy.distance import geodesic

def calcular_distancia(lat,long, merch_lat, merch_long):
    """
    Calcula la distancia en kilómetros entre las coordenadas de un comercio y un cliente.

    Args:
    lat: latitud domicilio cliente
    long: longitud domiclio cliente
    merch_lat: latitud comercio
    merch_long: longitud comercio
    
    Returns:
        float: Distancia en kilómetros.
    """
    cliente_coords = (lat, long)
    comercio_coords = (merch_lat, merch_long)
    distancia = geodesic(comercio_coords, cliente_coords).km
    return distancia

