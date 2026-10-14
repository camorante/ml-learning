import numpy as np
from resumen import media, media_ponderada

km = np.array([420.0, 380.0, 35.0, 25.0])          # dos viajes largos de autopista y dos repartos cortos
litros = np.array([126.0, 117.8, 15.4, 11.5])
l100 = 100 * litros / km                            # L/100 km de cada viaje
print("L/100 km por viaje:      ", l100.round(1))
print("media de los cocientes:  ", round(media(l100), 2))
print("litros totales / km × 100:", round(100 * litros.sum() / km.sum(), 2))
print("media ponderada por km:  ", round(media_ponderada(l100, km), 2))
