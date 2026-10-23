"""Datos de la versión sencilla: posición GPS del juguete, velocidad a 1 Hz y los 10 viajes del tema 4."""
import json, os
import numpy as np

# velocidad (km/h) = suma de escalones suaves S(u) = 3u² − 2u³; t en minutos
ESCALONES = [(0, 3, 50), (8, 2, 40), (19, 1, -60), (23, 2, 50)]     # (inicio, duración, salto)

def _S(u): return np.clip(3 * np.clip(u, 0, 1) ** 2 - 2 * np.clip(u, 0, 1) ** 3, 0, 1)
def _I(u):                                                          # ∫₀ᵘ S
    u = np.asarray(u, float); c = np.clip(u, 0, 1)
    return np.where(u <= 0, 0.0, c ** 3 - c ** 4 / 2 + np.maximum(u - 1, 0))

def velocidad(t): return sum(A * _S((t - t0) / d) for t0, d, A in ESCALONES)
def posicion(t): return sum(A * d * _I((t - t0) / d) for t0, d, A in ESCALONES) / 60    # km

_E = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "embed.json")))
VEL_1HZ = np.array(_E["v1"])                    # 61 lecturas de velocidad GPS (km/h), una por segundo
CARGA = np.array(_E["x"]); CONSUMO = np.array(_E["y"])   # tema 4: toneladas y L/100 km
