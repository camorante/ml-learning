"""Imágenes sintéticas para la idea 3 (comprimir): un camión dibujado y un mapa de actividad de la flota."""
import numpy as np


def camion(h=48, w=80):
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w))
    img[(yy >= 9) & (yy <= 33) & (xx >= 5) & (xx <= 50)] = 0.85          # caja
    cab = (yy >= 17) & (yy <= 33) & (xx >= 53) & (xx <= 71) & (xx - 53 <= (yy - 17) + 12)  # cabina con parabrisas inclinado
    img[cab] = 1.0
    win = (yy >= 19) & (yy <= 25) & (xx >= 56) & (xx <= 64) & (xx - 56 <= (yy - 19) + 6)
    img[win] = 0.25
    img[(yy >= 29) & (yy <= 31) & (xx >= 50) & (xx <= 53)] = 0.6          # enganche
    for cx in (15, 27, 62):
        r = np.hypot(yy - 35, xx - cx)
        img[r <= 5.5] = 1.0
        img[r <= 2.2] = 0.35
    img[(yy >= 42) & (yy <= 43)] = 0.45                                   # carretera
    img[(yy >= 42) & (yy <= 43) & ((xx // 6) % 2 == 0)] = 0.15
    return img


def actividad(seed=2):
    """Camiones en marcha (de 200) por hora del día (24 filas) y día del año (364 columnas, empieza en lunes)."""
    rng = np.random.default_rng(seed)
    h = np.arange(24)
    lab = np.exp(-0.5 * ((h - 10.5) / 3.6) ** 2) * (h >= 5) * (h <= 21) + 0.08
    sab = np.exp(-0.5 * ((h - 9.5) / 2.2) ** 2) * (h >= 6) * (h <= 15) * 0.55 + 0.05
    dom = 0.06 + 0.05 * np.exp(-0.5 * ((h - 20) / 2.0) ** 2)
    d = np.arange(364); dow = d % 7
    temporada = 1 + 0.10 * d / 364 - 0.35 * np.exp(-0.5 * ((d - 222) / 9) ** 2)   # crece en el año, baja en agosto
    temporada[(d >= 357)] *= 0.5
    perf = np.where(dow[None, :] < 5, lab[:, None], np.where(dow[None, :] == 5, sab[:, None], dom[:, None]))
    lam = 170 * perf * temporada[None, :]
    return rng.poisson(lam).astype(float)


def info(X, name):
    s = np.linalg.svd(X, compute_uv=False); e = np.cumsum(s**2) / (s**2).sum()
    m, n = X.shape
    print(name, X.shape, "energía acumulada k=1..8:", (100 * e[:8]).round(2))
    for k in (1, 2, 3, 5, 8, 12, 20):
        print(f"  k={k:2d} guardados {k * (m + n + 1):5d} de {m * n}  ({100 * k * (m + n + 1) / (m * n):.1f} %)  error rel. {np.sqrt(1 - e[k - 1]) * 100:.1f} %")

if __name__ == "__main__":
    info(camion(), "camión"); info(actividad(), "actividad")
