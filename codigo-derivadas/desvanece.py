import numpy as np

def normas(act, init, capas=20, ancho=64, seed=0):
    """‖∂L/∂h‖ a la entrada de cada capa de una red recién inicializada (media de 200 entradas)."""
    rng = np.random.default_rng(seed)
    W = [rng.normal(size=(ancho, ancho)) * init(ancho) for _ in range(capas)]
    h = rng.normal(size=(200, ancho)); zs = []
    for Wk in W:                                         # hacia delante
        z = h @ Wk.T; zs.append(z)
        h = 1 / (1 + np.exp(-z)) if act == "sigmoide" else np.maximum(z, 0)
    d = np.ones_like(h) / ancho                          # ∂L/∂h_salida para L = media de la salida
    out = []
    for k in range(capas - 1, -1, -1):                   # hacia atrás: d ← Wᵀ (d ⊙ act'(z))
        if act == "sigmoide":
            s = 1 / (1 + np.exp(-zs[k])); dz = d * s * (1 - s)        # σ' ≤ 0,25
        else:
            dz = d * (zs[k] > 0)                                       # ReLU' = 0 o 1
        d = dz @ W[k]; out.append(np.linalg.norm(d, axis=1).mean())
    return out[::-1]                                     # índice 0 = la primera capa

glorot = lambda n: np.sqrt(1 / n); he = lambda n: np.sqrt(2 / n)
for nombre, act, ini in (("sigmoide + Glorot", "sigmoide", glorot), ("ReLU + He", "relu", he),
                         ("ReLU, pesos × 1,5", "relu", lambda n: 1.5 * he(n))):
    r = normas(act, ini)
    print(f"{nombre:18s} capa 20: {r[19]:.2e}  capa 15: {r[14]:.2e}  capa 10: {r[9]:.2e}"
          f"  capa 1: {r[0]:.2e}  (1 / 20: {r[0] / r[19]:.1e})")
