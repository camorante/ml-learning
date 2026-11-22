"""Datos del tema 9: tramos del acelerómetro longitudinal de un camión (10 lecturas por segundo, en m/s²;
por debajo de 0 el camión frena), como en el tema 8, pero ahora de longitud variable y leídos paso a paso.

La pregunta de memoria: «¿hubo una frenada brusca al salir?». Cada tramo empieza con 16 lecturas (1,6 s) en las que
el camión frena, fuerte o suave, y sigue con un hueco de G lecturas de conducción normal: vibración, frenadas
suaves, baches y acelerones, nunca otra frenada brusca. La red da la respuesta al final, cuando ya ha leído todo.
La etiqueta es la regla de la flota del tema 8 aplicada a la señal limpia del arranque: brusca si en algún medio
segundo (5 lecturas) la deceleración media pasa de 3 m/s²."""
import numpy as np

FS = 10                     # lecturas por segundo
L0 = 16                     # lecturas del arranque, donde está la frenada que hay que recordar
UMBRAL = 3.0                # m/s² de deceleración media en 0,5 s (la regla del tema 8)
RUIDO = 0.35                # m/s²: vibración que mide el acelerómetro
HUECOS = (0, 10, 25, 50, 75, 100)   # lecturas entre el arranque y la pregunta


def media5(x):
    return np.convolve(x, np.ones(5) / 5, mode="valid")


def frenada(rng, prof, n_meseta):
    """Baja en 2–4 lecturas hasta −prof, se mantiene n_meseta lecturas y vuelve en 3–5 (como en el tema 8)."""
    a, c = rng.integers(2, 5), rng.integers(3, 6)
    return np.r_[-prof * np.arange(1, a + 1) / a, -prof * np.ones(n_meseta), -prof * np.arange(c - 1, -1, -1) / c]


def distractores(rng, limpia, desde):
    """Conducción normal en el hueco: de media un suceso cada 25 lecturas, frenada suave, bache o acelerón."""
    n = len(limpia) - desde
    for _ in range(rng.poisson(n / 25)):
        tipo = rng.integers(3)
        if tipo == 0:                                       # frenada suave: nunca llega a brusca
            f = frenada(rng, rng.uniform(0.8, 2.2), rng.integers(2, 8))
            if len(f) <= n:
                s = desde + rng.integers(0, n - len(f) + 1); limpia[s:s + len(f)] += f
        elif tipo == 1 and n >= 2:                          # bache: golpe de 2 lecturas, abajo y arriba
            b = desde + rng.integers(0, n - 1); A = rng.uniform(2.5, 5.0)
            limpia[b] -= A; limpia[b + 1] += A
        elif n >= 10:                                       # acelerón: positivo, largo y suave
            m = rng.integers(10, min(21, n + 1)); b = desde + rng.integers(0, n - m + 1)
            limpia[b:b + m] += rng.uniform(0.8, 2.0) * np.sin(np.pi * np.arange(m) / (m - 1))


def tramo(rng, G, brusca):
    """Un tramo de 16 + G lecturas. Devuelve (lectura con ruido, señal limpia, etiqueta)."""
    T = L0 + G
    t = np.arange(T)
    limpia = 0.25 * np.sin(2 * np.pi * (t / rng.uniform(40, 120) + rng.random())) + rng.normal(0, 0.1)
    while True:
        f = frenada(rng, rng.uniform(3.6, 5.5) if brusca else rng.uniform(1.0, 2.4), rng.integers(2, 6))
        if len(f) <= L0 - 1:
            break
    s = rng.integers(0, L0 - len(f) + 1)
    limpia[s:s + len(f)] += f
    base = limpia.copy()
    while True:                                             # el hueco nunca tiene otra frenada brusca
        limpia = base.copy(); distractores(rng, limpia, L0)
        if G < 5 or media5(limpia[L0:]).min() >= -UMBRAL:
            break
    y = int(media5(limpia[:L0]).min() < -UMBRAL)
    return limpia + rng.normal(0, RUIDO, T), limpia, y


def conjunto(n, G, seed):
    """n tramos con hueco G, mitad con frenada brusca al salir. X: (n, 16 + G), y: (n,)."""
    rng = np.random.default_rng(seed)
    V = [tramo(rng, G, k % 2 == 0) for k in range(n)]
    orden = rng.permutation(n)
    X = np.array([V[k][0] for k in orden]); Y = np.array([V[k][2] for k in orden])
    LIM = np.array([V[k][1] for k in orden])
    return X, Y, LIM


def datos(G):
    """Entrenamiento (2 000), validación (500) y prueba (1 000) para un hueco G, con semillas fijas."""
    return conjunto(2000, G, 900 + G), conjunto(500, G, 1900 + G), conjunto(1000, G, 2900 + G)


if __name__ == "__main__":
    for G in HUECOS:
        (X, Y, LIM), _, (Xt, Yt, _) = datos(G)
        regla = np.mean([(media5(x[:L0]).min() < -UMBRAL) == y for x, y in zip(Xt, Yt)])
        fin = np.mean([(media5(x[-L0:]).min() < -UMBRAL) == y for x, y in zip(Xt, Yt)]) if G >= L0 else float("nan")
        print(f"hueco {G:3d}: {X.shape[1]:3d} lecturas, bruscas {Y.mean():.1%} · regla sobre el arranque con ruido {regla:.1%}"
              f" · la misma regla sobre las 16 últimas {fin:.1%} · lectura mínima del hueco {LIM[:, L0:].min() if G else 0:.2f}")
