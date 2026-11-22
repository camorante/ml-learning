"""Datos del tema 10: órdenes de taller de una flota de camiones. En cada visita al taller, el lector de
diagnosis guarda los códigos de avería (DTC) que tenía el camión. Hay 20 códigos y cada uno pertenece a un
sistema (motor, escape, frenos, eléctrico), pero la red NO lo sabe: solo ve el número del código (0…19).
Una avería de un sistema suele encender varios códigos de ese sistema a la vez; tres códigos están a medio
camino entre dos sistemas (la pérdida de comunicación con una centralita, la válvula EGR).

De cada visita salen pares (código visto, otro código de la misma visita): la tarea de predicción con la que
se aprenden los embeddings, como en word2vec, pero con códigos de avería en lugar de palabras."""
import numpy as np

SISTEMAS = ["motor", "escape", "frenos", "eléctrico"]
# código, qué es, peso en cada sistema (motor, escape, frenos, eléctrico), popularidad
CODIGOS = [
    ("P0217", "sobretemperatura del motor",          (1, 0, 0, 0), 1.2),
    ("P0128", "termostato: el motor no se calienta", (1, 0, 0, 0), 0.9),
    ("P0524", "presión de aceite baja",              (1, 0, 0, 0), 0.7),
    ("P0087", "presión de combustible baja",         (1, 0, 0, 0), 1.0),
    ("P0299", "turbo: poca presión de soplado",      (1, 0, 0, 0), 1.3),
    ("P2002", "filtro de partículas: eficiencia baja", (0, 1, 0, 0), 1.3),
    ("P2463", "filtro de partículas: mucho hollín",  (0, 1, 0, 0), 1.1),
    ("P20EE", "catalizador SCR: eficiencia baja",    (0, 1, 0, 0), 1.0),
    ("P20B9", "calentador de AdBlue",                (0, 1, 0, 0), 0.6),
    ("P0401", "válvula EGR: poco caudal",            (0.45, 0.55, 0, 0), 1.0),
    ("C0035", "sensor de rueda delantera izquierda", (0, 0, 1, 0), 1.1),
    ("C0040", "sensor de rueda delantera derecha",   (0, 0, 1, 0), 1.0),
    ("C0110", "bomba del ABS",                       (0, 0, 1, 0), 0.7),
    ("C0265", "relé de la válvula del ABS",          (0, 0, 1, 0), 0.8),
    ("C0561", "control de estabilidad desactivado",  (0, 0, 1, 0), 0.9),
    ("P0562", "tensión del sistema baja",            (0, 0, 0, 1), 1.3),
    ("P0620", "alternador: circuito de control",     (0, 0, 0, 1), 0.8),
    ("B1325", "batería: tensión baja al arrancar",   (0, 0, 0, 1), 1.0),
    ("U0100", "sin comunicación con la centralita del motor", (0.45, 0, 0, 0.55), 0.9),
    ("U0121", "sin comunicación con el ABS",         (0, 0, 0.45, 0.55), 0.9),
]
V = len(CODIGOS)
NOMBRES = [c[0] for c in CODIGOS]
DESC = [c[1] for c in CODIGOS]
PERT = np.array([c[2] for c in CODIGOS], dtype=float)          # (20, 4)
POP = np.array([c[3] for c in CODIGOS])
SISTEMA = PERT.argmax(1)                                        # el sistema principal de cada código
PI = np.array([0.30, 0.22, 0.26, 0.22])                         # qué sistema falla en cada visita
RUIDO = 0.04                                                    # códigos sueltos que no tienen que ver


def visitas(n, seed):
    """n órdenes de taller: se elige el sistema averiado y se encienden 2, 3 o 4 códigos distintos."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = rng.choice(4, p=PI)
        k = rng.choice([2, 3, 4], p=[0.45, 0.35, 0.20])
        p = POP * PERT[:, s] + RUIDO
        out.append(sorted(rng.choice(V, size=k, replace=False, p=p / p.sum()).tolist()))
    return out


def pares(vs):
    """Todos los pares ordenados (a, b) con a ≠ b de cada visita: «visto a, ¿qué otro código hay?»."""
    P = [(a, b) for v in vs for a in v for b in v if a != b]
    return np.array(P, dtype=np.int64)


VIS_TR = visitas(400, 10)          # entrenamiento
VIS_TE = visitas(1000, 11)         # prueba
P_TR, P_TE = pares(VIS_TR), pares(VIS_TE)

if __name__ == "__main__":
    print(f"{V} códigos, {len(VIS_TR)} visitas de entrenamiento ({len(P_TR)} pares), {len(VIS_TE)} de prueba ({len(P_TE)} pares)")
    print("primeras visitas:", [[NOMBRES[c] for c in v] for v in VIS_TR[:4]])
    cnt = np.bincount(P_TR[:, 0], minlength=V)
    print("veces que aparece cada código como «visto» (entrenamiento):", dict(zip(NOMBRES, cnt.tolist())))
