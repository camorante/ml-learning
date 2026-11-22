"""Datos del tema 13: una flota simulada de 3 000 camiones durante 2 años, con lecturas diarias de tres sensores del
motor y sus averías. Cada fin de mes (24 «fotos» por camión) se pregunta: ¿se averiará en los próximos 30 días?

Entradas de cada foto:
  · secuencia: los últimos 28 días de tres sensores (vibración en mm/s, temperatura máxima del refrigerante en °C y
    presión de aceite en bar);
  · tabla: edad (años), kilómetros totales (miles), carga media del mes (t), kilómetros del mes (miles) y averías
    del camión en los 12 meses anteriores (solo pasado);
  · categorías: modelo de camión (8) y depósito (10).
Lo que la red no ve: la fragilidad propia de cada camión y si una avería avisó antes en los sensores."""
import os
import numpy as np

N_CAM, MESES, VENT, HOR = 3000, 24, 28, 30           # camiones, fotos por camión, días de secuencia, días de horizonte
HIST = 365                                            # un año de historial antes de la primera foto
DIAS = HIST + VENT + 30 * (MESES - 1) + HOR           # 1 113 días simulados
MODELOS = ["A", "B", "C", "D", "E", "F", "G", "H"]
DEPOS = ["Madrid", "Barcelona", "Valencia", "Sevilla", "Zaragoza", "Bilbao", "Asturias", "León", "Málaga", "Murcia"]
MONTANA = [6, 7]                                      # Asturias y León: puertos de montaña
LARGO = [0, 4, 7, 9]                                  # depósitos de larga distancia (más km al año)
EF_MOD = np.array([0.0, 0.3, -0.2, 0.5, 0.1, -0.3, 0.4, 0.0])   # fragilidad de cada modelo (log del riesgo)
CANALES = ["vibración", "temperatura", "presión de aceite"]
SIG = np.array([0.4, 1.2, 0.15])                       # ruido diario de cada sensor (mm/s, °C, bar)
P_AVISO = 0.75                                         # probabilidad de que una avería avise en los sensores
B0 = -7.36                                             # nivel de riesgo: ≈ 8 % de fotos con avería en 30 días


def simular(seed=13):
    rng = np.random.default_rng(seed)
    n = N_CAM
    mod = rng.choice(8, n, p=[0.16, 0.14, 0.14, 0.12, 0.12, 0.12, 0.10, 0.10])
    dep = rng.integers(0, 10, n)
    edad0 = rng.uniform(0.5, 10, n)
    km_an = np.where(np.isin(dep, LARGO), 140, 75) * rng.normal(1, 0.12, n)          # miles de km al año
    carga = np.clip(rng.normal(18, 3.5, n) + 2 * np.isin(dep, LARGO), 8, 26)        # t
    frag = rng.normal(0, 1.2, n)                                                     # fragilidad propia (oculta)
    km0 = edad0 * km_an * rng.uniform(0.8, 1.2, n)
    t = np.arange(DIAS)
    edad = edad0[:, None] + t / 365
    mont = np.isin(dep, MONTANA)
    inter = 1.0 * ((mod == 5) & mont)                                                 # modelo F en la montaña: se calienta
    eta = (B0 + 0.09 * edad + 0.25 * (km_an[:, None] / 100 - 1) + 0.05 * (carga[:, None] - 18)
           + EF_MOD[mod][:, None] + 0.3 * mont[:, None] + inter[:, None] + frag[:, None])
    h = np.exp(eta)                                                                   # riesgo diario
    # averías: un día con probabilidad h; tras una avería, 15 días sin otra (reparación)
    u = rng.random((n, DIAS))
    fallos = []
    for i in range(n):
        f, ult = [], -99
        for d in np.flatnonzero(u[i] < h[i]):
            if d - ult >= 15:
                f.append(d); ult = d
        fallos.append(f)
    # tipo de avería: 0 rodamiento (vibración), 1 refrigeración (temperatura), 2 aceite (presión)
    sens = np.zeros((n, DIAS, 3))
    nivel = np.c_[rng.normal(4.0, 0.5, n) + 0.15 * mod, rng.normal(88, 1.5, n) + 0.1 * (carga - 18), rng.normal(4.0, 0.2, n)]
    estacion = 2.5 * np.sin(2 * np.pi * (t - 100) / 365)                              # verano más caluroso
    w = np.zeros((n, 3)); deriva = np.zeros((n, DIAS, 3))
    for d in range(DIAS):                                                             # vaivén lento AR(1)
        w = 0.85 * w + rng.normal(0, 0.35, (n, 3)); deriva[:, d] = w
    sens = nivel[:, None, :] + SIG * (deriva + rng.normal(0, 1, (n, DIAS, 3)))
    sens[:, :, 1] += estacion
    # molestias sin avería: baches (vibración), olas de calor (temperatura) y fallos del sensor de aceite
    bache = rng.random((n, DIAS)) < 0.05
    sens[:, :, 0] += bache * rng.uniform(1.5, 3.5, (n, DIAS)) * SIG[0] * 2.5
    calor = rng.random((n, DIAS)) < 0.01
    for k in range(4):
        sens[:, k:, 1] += np.roll(calor, k, 1)[:, k:] * 1.8 * SIG[1] * 1.5
    glitch = rng.random((n, DIAS)) < 0.02
    sens[:, :, 2] -= glitch * rng.uniform(2, 4, (n, DIAS)) * SIG[2]
    info = []                                                                         # (camión, día, tipo, aviso, días de aviso)
    for i in range(n):
        for F in fallos[i]:
            tipo = 1 if (inter[i] and rng.random() < 0.7) else rng.choice(3, p=[0.4, 0.35, 0.25])
            avisa = rng.random() < P_AVISO
            L = int(rng.integers(4, 36)) if avisa else 0
            info.append((i, F, tipo, int(avisa), L))
            if not avisa:
                continue
            a = max(0, F - L)
            prog = (np.arange(a, F) - (F - L) + 1) / L                                # de 0 a 1 hasta la avería
            if tipo == 0:
                sens[i, a:F, 0] += SIG[0] * (1.5 * prog + (rng.random(F - a) < 0.15 + 0.55 * prog) * rng.uniform(2.5, 5, F - a))
            elif tipo == 1:
                sens[i, a:F, 1] += SIG[1] * (2.5 * prog + (rng.random(F - a) < 0.2) * 1.5)
            else:
                sens[i, a:F, 2] -= SIG[2] * 2.8 * prog
    # fotos de fin de mes
    S = HIST + VENT + 30 * np.arange(MESES)                                           # día de cada foto
    previas = np.array([[sum(s - 365 <= F < s for F in fallos[i]) for s in S] for i in range(n)])
    X = np.stack([sens[:, s - VENT:s] for s in S], 1).astype(np.float32)              # (n, 24, 28, 3)
    tab = np.stack([np.stack([edad0 + s / 365, (km0 + km_an * s / 365), carga + rng.normal(0, 1, n),
                              km_an / 12 * rng.normal(1, 0.1, n), previas[:, m]], 1) for m, s in enumerate(S)], 1)  # (n, 24, 5)
    Y = np.zeros((n, MESES), np.int8); VIS = np.full((n, MESES), -1, np.int8); TIPO = np.full((n, MESES), -1, np.int8)
    for (i, F, tipo, avisa, L) in info:
        for m, s in enumerate(S):
            if s <= F < s + HOR and Y[i, m] == 0:
                Y[i, m] = 1; TIPO[i, m] = tipo
                VIS[i, m] = int(avisa and F - L < s)                                   # el aviso ya había empezado
    return dict(X=X, tab=tab.astype(np.float32), mod=mod, dep=dep, Y=Y, VIS=VIS, TIPO=TIPO, frag=frag, ntot=np.array([len(f) for f in fallos]),
                fallos=np.array(info), S=S)


def reparto(seed=7):
    """Por camión (60/20/20) y por tiempo: entrena con los meses 0–15, valida con 16–19 y examina con 20–23."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(N_CAM)
    return dict(tr=(np.sort(p[:1800]), np.arange(0, 16)), va=(np.sort(p[1800:2400]), np.arange(16, 20)),
                te=(np.sort(p[2400:]), np.arange(20, 24)))


_RUTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos.npz")


def cargar():
    if not os.path.exists(_RUTA):
        np.savez_compressed(_RUTA, **simular())
    z = np.load(_RUTA)
    return {k: z[k] for k in z.files}


def filas(D, parte, R=None):
    """Aplana (camiones, meses) de un trozo del reparto en filas: secuencia, tabla, modelo, depósito, etiqueta,
    camión, mes, aviso visible y tipo."""
    R = R or reparto()
    cam, mes = R[parte]
    C, M = np.meshgrid(cam, mes, indexing="ij"); C, M = C.ravel(), M.ravel()
    return dict(X=D["X"][C, M], tab=D["tab"][C, M], mod=D["mod"][C], dep=D["dep"][C], y=D["Y"][C, M].astype(np.float32),
                cam=C, mes=M, vis=D["VIS"][C, M], tipo=D["TIPO"][C, M])


if __name__ == "__main__":
    D = cargar()
    Y = D["Y"]
    print(f"{N_CAM} camiones × {MESES} fotos = {Y.size} fotos; con avería en 30 días: {Y.mean():.1%}")
    f = D["fallos"]
    print(f"averías: {len(f)}; avisan en los sensores: {f[:, 3].mean():.1%}; tipos (rodamiento, refrigeración, aceite):",
          np.bincount(f[:, 2], minlength=3))
    pos = D["VIS"][Y == 1]
    print(f"fotos con avería cuyo aviso ya se ve en los 28 días: {pos.mean():.1%}")
    R = reparto()
    for k, nm in [("tr", "entrenamiento"), ("va", "validación"), ("te", "examen")]:
        F = filas(D, k, R)
        print(f"{nm:13s} {len(R[k][0])} camiones × {len(R[k][1])} meses = {len(F['y'])} fotos, con avería {F['y'].mean():.1%}")
