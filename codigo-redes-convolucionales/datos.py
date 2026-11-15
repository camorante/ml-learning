"""Datos del tema 8: ventanas de 6,4 s del acelerómetro longitudinal de un camión (10 lecturas por segundo,
64 lecturas, en m/s²; por debajo de 0 el camión frena). Cada ventana tiene una frenada, suave o brusca,
y a veces un bache o un acelerón. La etiqueta viene de la velocidad del bus CAN, que es precisa: la frenada
es «brusca» si en algún medio segundo (5 lecturas) la deceleración media pasa de 3 m/s². La red solo ve el
acelerómetro barato, que añade la vibración de la carretera."""
import numpy as np

FS, L = 10, 64              # lecturas por segundo y lecturas por ventana
UMBRAL = 3.0                # m/s² de deceleración media en 0,5 s: la regla de «frenada brusca» de la flota
RUIDO = 0.35                # m/s²: vibración de la carretera y del motor que mide el acelerómetro


def media5(x):
    """Media móvil de 5 lecturas (medio segundo)."""
    return np.convolve(x, np.ones(5) / 5, mode="valid")


def frenada(rng, prof, n_meseta):
    """Forma de una frenada: baja en 2–4 lecturas hasta −prof, se mantiene n_meseta y vuelve en 3–5."""
    a, c = rng.integers(2, 5), rng.integers(3, 6)
    return np.r_[-prof * np.arange(1, a + 1) / a, -prof * np.ones(n_meseta), -prof * np.arange(c - 1, -1, -1) / c]


def ventana(rng, zona):
    """Una ventana: deriva lenta + frenada (en la zona de inicio dada) + quizá bache y acelerón + ruido.
    Devuelve (lectura del acelerómetro, señal limpia, etiqueta, inicio de la frenada)."""
    t = np.arange(L)
    limpia = 0.25 * np.sin(2 * np.pi * (t / rng.uniform(40, 120) + rng.random())) + rng.normal(0, 0.1)
    f = frenada(rng, rng.uniform(1.0, 5.5), rng.integers(2, 13))
    lo, hi = zona
    s = rng.integers(lo, min(hi, L - len(f)) + 1)
    limpia[s:s + len(f)] += f
    if rng.random() < 0.6:                                  # bache: golpe de 2 lecturas, abajo y arriba
        b = rng.integers(1, L - 2); A = rng.uniform(2.5, 5.0)
        limpia[b] -= A; limpia[b + 1] += A
    if rng.random() < 0.4:                                  # acelerón: positivo, largo y suave
        n = rng.integers(10, 21); b = rng.integers(0, L - n + 1)
        limpia[b:b + n] += rng.uniform(0.8, 2.0) * np.sin(np.pi * np.arange(n) / (n - 1))
    y = int(media5(limpia).min() < -UMBRAL)
    return limpia + rng.normal(0, RUIDO, L), limpia, y, int(s)


def conjunto(n, seed, zona=(1, L)):
    rng = np.random.default_rng(seed)
    V = [ventana(rng, zona) for _ in range(n)]
    return (np.array([v[0] for v in V]), np.array([v[2] for v in V]), np.array([v[1] for v in V]),
            np.array([v[3] for v in V]))


# A: grabación continua, la frenada puede estar en cualquier sitio de la ventana
X_TR, Y_TR, LIMPIA_TR, INI_TR = conjunto(2000, seed=81)
X_VA, Y_VA, LIMPIA_VA, INI_VA = conjunto(1000, seed=82)
X_TE, Y_TE, LIMPIA_TE, INI_TE = conjunto(1000, seed=83)
# B: el equipo antiguo grababa al detectar una frenada, así que la frenada empieza siempre al principio
PRINCIPIO, FINAL = (2, 20), (38, 64)
XB_TR, YB_TR, _, INIB_TR = conjunto(2000, seed=84, zona=PRINCIPIO)
XB_VA, YB_VA, _, _ = conjunto(1000, seed=85, zona=PRINCIPIO)
XB_FI, YB_FI, _, INIB_FI = conjunto(1000, seed=86, zona=FINAL)      # prueba: la frenada al final

if __name__ == "__main__":
    for nm, y in [("A entrenamiento", Y_TR), ("A validación", Y_VA), ("A prueba", Y_TE), ("B entrenamiento", YB_TR),
                  ("B al final", YB_FI)]:
        print(f"{nm:16s} {len(y)} ventanas, frenadas bruscas {y.mean():.1%}")
    print("inicio de la frenada en B:", INIB_TR.min(), INIB_TR.max(), " al final:", INIB_FI.min(), INIB_FI.max())
    m = np.array([media5(x).min() for x in X_VA])
    print("regla aplicada al acelerómetro con ruido (sin red):", f"{np.mean((m < -UMBRAL) == Y_VA):.1%}")
    print("regla de una sola lectura (mínimo < −3):", f"{np.mean((X_VA.min(1) < -UMBRAL) == Y_VA):.1%}")
