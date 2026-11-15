"""Comparación de regularizadores en los datos de flota: red 1-64-64-1, 20 viajes, 3 000 épocas de Adam/AdamW (tasa 0.01,
lote completo), 5 semillas (pesos iniciales y máscaras). Error de validación (RMSE, L/100 km) al final y en la mejor época
(parada temprana). La prueba se mira UNA vez, solo con la receta elegida por validación."""
import numpy as np
from datos import V_TR, Y_TR, V_VA, Y_VA, V_TE, Y_TE, ex, ey, rmse, verdad, SY
from mlp import entrenar, adelante

CFG = {"sin regularizar": {}, "Adam + L2 (λ = 0.01)": dict(wd=0.01, desacoplado=False),
       "Adam + L2 (λ = 0.1)": dict(wd=0.1, desacoplado=False), "AdamW (λ = 1)": dict(wd=1),
       "dropout 0.2": dict(p=0.2), "AdamW (λ = 1) + dropout 0.2": dict(wd=1, p=0.2)}
X, y, Xv, yv = ex(V_TR), ey(Y_TR), ex(V_VA), ey(Y_VA)
rm = lambda m: float(np.sqrt(m) * SY)

if __name__ == "__main__":
    print(f"{'':30s}  validación al final   con parada temprana   época elegida   entrenamiento al final")
    res = {}
    for k, c in CFG.items():
        R = [entrenar(X, y, Xv, yv, epocas=3000, lr=0.01, seed_init=s, seed_drop=s + 1, **c) for s in range(5)]
        vf = [rm(r["va"][-1]) for r in R]; vm = [rm(r["va"][r["mejor"]]) for r in R]; tf = [rm(r["tr"][-1]) for r in R]
        res[k] = (np.mean(vm), R)
        print(f"{k:30s}    {np.mean(vf):.3f} ± {np.std(vf):.3f}       {np.mean(vm):.3f} ± {np.std(vm):.3f}"
              f"       {int(np.median([r['mejor'] for r in R])):5d}          {np.mean(tf):.3f}")
    elegido = min(res, key=lambda k: res[k][0])
    r = res[elegido][1][0]                                          # semilla 0, pesos de su mejor época
    pv = adelante(r["Pm"], Xv)[0]
    e = pv * SY + (Y_TR.mean()) - Y_VA                              # errores de validación en L/100 km
    rng = np.random.default_rng(0)
    B = [np.sqrt(np.mean(e[rng.integers(0, 30, 30)] ** 2)) for _ in range(10000)]
    print(f"elegido por validación: {elegido} (semilla 0, época {r['mejor']}): validación {rmse(pv, Y_VA):.3f},"
          f" intervalo bootstrap del 95 % con 30 viajes: {np.percentile(B, 2.5):.2f}–{np.percentile(B, 97.5):.2f}")
    print(f"prueba, mirada una vez (200 viajes del mes siguiente): {rmse(adelante(r['Pm'], ex(V_TE))[0], Y_TE):.3f}"
          f"   ·   la curva de verdad daría {np.sqrt(np.mean((verdad(V_TE) - Y_TE) ** 2)):.3f} (solo el ruido)")
