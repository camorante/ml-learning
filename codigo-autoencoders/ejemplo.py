"""Ejemplo resuelto: un instante de 4 sensores del motor por un autoencoder lineal 4 → 2 → 4 con pesos atados
(el decodificador usa los mismos pesos traspuestos). Sus pesos son las 2 direcciones del PCA de 2 000 instantes
normales, redondeadas a 2 decimales; el umbral, el percentil 99 del error en 1 000 instantes normales de validación."""
import numpy as np
from sklearn.decomposition import PCA
from modelos import entrenar, reconstruir

NOMBRES = ["turbo (bar)", "presión de aceite (bar)", "temp. de aceite (°C)", "temp. de refrigerante (°C)"]


def instantes(n, seed):
    """carga y calor del motor, al azar; cada sensor responde a los dos, más su propio ruido."""
    rng = np.random.default_rng(seed)
    c, h = rng.normal(size=(2, n))
    return np.c_[1.4 + 0.5 * (0.9 * c + 0.1 * h) + rng.normal(0, 0.15, n),
                 3.5 + 0.6 * (0.8 * c - 0.3 * h) + rng.normal(0, 0.15, n),       # aceite caliente: menos espeso, menos presión
                 95 + 6 * (0.4 * c + 0.8 * h) + rng.normal(0, 1.5, n),
                 88 + 4 * (0.3 * c + 0.9 * h) + rng.normal(0, 1.0, n)]


TR, VA = instantes(2000, 1201), instantes(1000, 1202)
mu, sd = TR.mean(0).round(2), TR.std(0).round(2)                 # tipificar con la media y la desviación, redondeadas
pca = PCA(2).fit((TR - mu) / sd)
W = pca.components_.round(2)                                      # (2, 4): los pesos del codificador
W *= np.sign(W[:, 0:1])                                           # signo: que el turbo pese en positivo en las dos
print("media:", mu, " desviación:", sd)
print("pesos del codificador W (2 × 4):\n", W)
print(f"el PCA explica {pca.explained_variance_ratio_.sum():.1%} de la variación de los 4 sensores tipificados")


def error(X):
    x = (X - mu) / sd
    return (((x @ W.T) @ W - x) ** 2).mean(1)


u = float(np.quantile(error(VA), 0.99))
print(f"umbral (percentil 99 en validación) = {u:.4f}")

for nombre, raw in [("bomba de aceite gastada", np.array([2.0, 3.1, 100.0, 91.0])),
                    ("instante normal", VA[0].round(2))]:
    x = (raw - mu) / sd
    z = W @ x
    xr = W.T @ z
    r = xr - x
    e = np.mean(r ** 2)
    print(f"\n{nombre}: lectura {raw}")
    print("  tipificada x =", x.round(4))
    print("  código z = W·x =", z.round(4))
    print("  reconstrucción x̂ = Wᵀ·z =", xr.round(4), " → en unidades:", (xr * sd + mu).round(2))
    print("  residuo x̂ − x =", r.round(4), " cuadrados:", (r ** 2).round(4))
    print(f"  error = {e:.4f} → {'ALARMA' if e > u else 'normal'} (umbral {u:.4f}, {e / u:.1f} veces)")

lin, _ = entrenar((TR - mu) / sd, k=2, m=0, lineal=True, epocas=300, lr=1e-2, lote=32, seed=0)
x = (np.array([2.0, 3.1, 100.0, 91.0]) - mu) / sd
print(f"\ncomprobación: un autoencoder lineal 4 → 2 → 4 entrenado (sin atar ni redondear) da a la bomba gastada"
      f" un error de {((reconstruir(lin, x) - x) ** 2).mean():.4f}")
