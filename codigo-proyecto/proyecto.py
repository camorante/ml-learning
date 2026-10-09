"""El proyecto completo: limpiar, preparar variables, entrenar, elegir el límite y evaluar."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CFN, CFP = 4000, 150
T_COSTE = CFP / (CFP + CFN)
BASE = ["edad", "km", "horas_mant", "vibracion", "temp_motor", "codigos"]

def limpiar(d):
    d = d.drop_duplicates()                                                   # reenvíos idénticos (tema 3)
    t = d.temp_motor.where(d.temp_motor.between(40, 130))                      # centinelas -40 y 255 → hueco (temas 3 y 4)
    return d.assign(temp_motor=t)

def variables(d, fuga=False, indicador=True):
    d = d.sort_values(["camion", "semana"], kind="mergesort").copy(); g = d.groupby("camion")
    cols = list(BASE)
    if indicador:
        d["vib_falta"] = d.vibracion.isna().astype(float); cols.append("vib_falta")             # el hueco avisa (tema 5)
    # historial del camión, solo hacia atrás (tema 7). Con fuga=True la ventana incluye la propia semana,
    # cuya avería todavía no se conoce el lunes en que hay que decidir.
    desfase = 0 if fuga else 1
    d["averias_12s"] = g.averia_sig.transform(lambda s: s.shift(desfase).rolling(12, min_periods=1).sum()).fillna(0)
    d["vib_4s"] = g.vibracion.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    d["vib_cambio"] = d.vibracion - d.vib_4s
    d["cod_4s"] = g.codigos.transform(lambda s: s.shift(1).rolling(4, min_periods=1).sum()).fillna(0)
    cols += ["averias_12s", "vib_4s", "vib_cambio", "cod_4s"]
    return d.sort_index(), cols

def modelo(equilibrar=False):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=50,
                                          class_weight="balanced" if equilibrar else None, random_state=0)

def coste(y, p, t):
    a = p >= t; fn = int(np.sum(~a & y)); fp = int(np.sum(a & ~y))
    return fn * CFN + fp * CFP, fn, fp

def preparar(X, cols, medias, indicador):
    X = X[cols].copy()
    if not indicador: X = X.fillna(medias)                                   # rellenar con la media, sin indicador
    return X
