"""Datos del tema 12: ventanas de 6,4 s del acelerómetro vertical de la cabina de un camión (10 lecturas por segundo,
64 lecturas, en m/s², sin la gravedad). En marcha normal la cabina rebota sobre su suspensión a su frecuencia propia
(de 1,1 a 1,5 Hz según la carga), con más o menos fuerza según el firme, y la carretera añade una vibración fina.
Las ventanas «normales» son las únicas que ve el autoencoder al entrenar. Para probarlo inyectamos cuatro averías:
  atascado   el sensor se congela: desde un instante, repite la misma lectura
  pico       un golpe eléctrico en el cable: una sola lectura de 2,5 a 5 m/s²
  deriva     el sensor se descalibra: un desplazamiento que crece en línea recta durante la ventana
  vibración  una pieza que falla (un cardán desequilibrado): una oscilación rápida de 3,3 Hz, 3 lecturas por ciclo"""
import numpy as np

FS, L = 10, 64                  # lecturas por segundo y lecturas por ventana
RUIDO = 0.15                    # m/s²: vibración fina de la carretera
FALSA, PERDIDA = 80, 1500       # €: revisar en el taller un camión que estaba bien · una avería que se escapa
TIPOS = ["normal", "atascado", "pico", "deriva", "vibración"]
t = np.arange(L)


def normal(rng):
    """Rebote de la cabina: amplitud según el firme, frecuencia propia según la carga, fase al azar; más el ruido."""
    A, f, fase = rng.uniform(0.3, 1.2), rng.uniform(1.1, 1.5), rng.uniform(0, 2 * np.pi)
    limpia = A * np.sin(2 * np.pi * f * t / FS + fase)
    return limpia, limpia + rng.normal(0, RUIDO, L), (A, f, fase)


def averia(rng, tipo):
    """Una ventana normal con la avería del tipo dado. Devuelve (lectura, limpia sin avería)."""
    limpia, x, _ = normal(rng)
    x = x.copy()
    if tipo == "atascado":
        s = rng.integers(16, 49); x[s:] = x[s]
    elif tipo == "pico":
        b = rng.integers(2, L - 2); x[b] += rng.choice([-1, 1]) * rng.uniform(2.5, 5.0)
    elif tipo == "deriva":
        x += rng.choice([-1, 1]) * rng.uniform(0.6, 2.0) * t / (L - 1)
    elif tipo == "vibración":
        x += rng.uniform(0.15, 0.6) * np.sin(2 * np.pi * (10 / 3) * t / FS + rng.uniform(0, 2 * np.pi))
    return x, limpia


def normales(n, seed, parametros=False):
    rng = np.random.default_rng(seed)
    V = [normal(rng) for _ in range(n)]
    X = np.array([v[1] for v in V])
    return (X, np.array([v[2] for v in V])) if parametros else X


def prueba(n_normal, n_cada, seed):
    """Conjunto de prueba: n_normal ventanas normales y n_cada de cada avería, mezcladas. Devuelve (X, tipo)."""
    rng = np.random.default_rng(seed)
    X, y = [normal(rng)[1] for _ in range(n_normal)], [0] * n_normal
    for k in range(1, 5):
        X += [averia(rng, TIPOS[k])[0] for _ in range(n_cada)]; y += [k] * n_cada
    return np.array(X), np.array(y)


X_TR = normales(3000, seed=121)                 # entrenamiento: solo marcha normal
X_VA = normales(1000, seed=122)                 # validación normal: para elegir el umbral
X_TE, Y_TE = prueba(2000, 50, seed=123)         # prueba: 2 000 normales + 50 de cada avería

if __name__ == "__main__":
    print(f"entrenamiento {X_TR.shape}, validación {X_VA.shape}, prueba {X_TE.shape}")
    print("prueba por tipo:", {TIPOS[k]: int((Y_TE == k).sum()) for k in range(5)})
    print(f"desviación típica de una lectura normal: {X_TR.std():.3f} m/s²; del ruido: {RUIDO}")
    print(f"error cuadrático medio si se reconstruye la señal limpia perfecta: {RUIDO ** 2:.4f}")
