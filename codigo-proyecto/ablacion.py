import numpy as np, itertools, json
from sim import flota
from proyecto import *
from sklearn.model_selection import train_test_split

bruto = flota()
verdad = bruto.drop_duplicates()                                        # para medir la realidad: cada camión-semana una vez
def correr(limpio, indicador, sin_fuga, sin_pesos, umbral_coste):
    d0 = limpiar(bruto) if limpio else bruto.copy()
    d, cols = variables(d0, fuga=not sin_fuga, indicador=indicador)
    prod, _ = variables(d0, fuga=False, indicador=indicador)              # en producción la semana actual no se conoce
    dev = d[d.semana < 60]; tr, ev = dev[dev.semana < 48], dev[dev.semana >= 48]   # examen: semanas 48-59 (tema 1)
    medias = tr[cols].mean()
    m = modelo(equilibrar=not sin_pesos).fit(preparar(tr, cols, medias, indicador), tr.averia_sig)
    t = T_COSTE if umbral_coste else 0.5
    c, _, _ = coste(ev.averia_sig.values, m.predict_proba(preparar(ev, cols, medias, indicador))[:, 1], t)
    informe = c / len(ev) * 300
    fut = prod[(prod.semana >= 60) & (d.semana < 80)].drop_duplicates(subset=["semana", "camion"])      # la realidad: semanas 60-79
    c2, fn, fp = coste(fut.averia_sig.values, m.predict_proba(preparar(fut, cols, medias, indicador))[:, 1], t)
    return round(informe), round(c2 / len(fut) * 300), fn, fp
R = {}
for k in itertools.product([1, 0], repeat=5):
    R["".join(map(str, k))] = correr(*k); print(k, R["".join(map(str, k))])
fut = verdad[(verdad.semana >= 60) & (verdad.semana < 80)]
R["nunca"] = round(fut.averia_sig.sum() * CFN / len(fut) * 300)
print("nunca", R["nunca"])
json.dump(R, open("../ablacion.json", "w"))
