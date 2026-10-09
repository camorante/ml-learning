"""El modelo en producción, semanas 60-103: vigilancia y tres reacciones al cambio de sensor de la semana 80."""
import numpy as np, json
from sim import flota
from proyecto import *
from monitor import psi, ks

bruto = flota(); d0 = limpiar(bruto)
def entrenar(d0, hasta):
    d, cols = variables(d0); tr = d[d.semana < hasta]
    return modelo().fit(tr[cols], tr.averia_sig), cols, tr
m0, cols, tr0 = entrenar(d0, 60)                                     # el modelo que se despliega: semanas 0-59
ref = tr0.vibracion.values

def semanal(d0, modelo_por_semana):
    d, _ = variables(d0); out = []
    for s in range(60, 104):
        w = d[d.semana == s]; m = modelo_por_semana(s)
        p = m.predict_proba(w[cols])[:, 1]; c, fn, fp = coste(w.averia_sig.values, p, T_COSTE)
        out.append(dict(semana=s, coste=c, avisos=int((p >= T_COSTE).sum()), averias=int(w.averia_sig.sum()), fn=fn, fp=fp,
                        psi=round(psi(ref, w.vibracion.values), 4), ks=round(ks(ref[~np.isnan(ref)], w.vibracion.dropna().values), 4),
                        p_media=round(float(p.mean()), 4)))
    return out

A = semanal(d0, lambda s: m0)                                          # 1) no hacer nada
# 2) reentrenar: la alarma salta al cerrar la semana 80 (PSI > 0,25). Cada 4 semanas se reentrena con todo lo que
#    tiene etiqueta (la avería de la semana s se conoce al terminar la s+1).
cache = {}
def reentrenado(s):
    if s <= 81: return m0
    h = 82 + 4 * ((s - 82) // 4)
    if h not in cache: cache[h] = entrenar(d0, h - 1)[0]
    return cache[h]
B = semanal(d0, reentrenado)
# 3) arreglar el dato: para cada camión, cociente entre la mediana de vibración de las semanas 81-84 y la de 76-79.
#    Los que bajan más de un 20 % llevan el sensor nuevo: se divide su lectura por ese cociente desde la semana 80.
#    (Hasta la semana 85 no hay datos para estimarlo: esas semanas siguen igual que sin hacer nada.)
med_a = d0[d0.semana.between(76, 79)].groupby("camion").vibracion.median(); med_d = d0[d0.semana.between(81, 84)].groupby("camion").vibracion.median()
ratio = (med_d / med_a); nuevos = ratio[ratio < 0.8]; factor = float(nuevos.median())
print("camiones con sensor nuevo detectados", len(nuevos), "factor", round(factor, 3))
d_fix = d0.copy(); msk = d_fix.camion.isin(nuevos.index) & (d_fix.semana >= 80)
d_fix.loc[msk, "vibracion"] = d_fix.loc[msk, "vibracion"] / factor
Cfix = semanal(d_fix, lambda s: m0)
C = [a if a["semana"] < 85 else c for a, c in zip(A, Cfix)]
nunca = [dict(semana=r["semana"], coste=r["averias"] * CFN) for r in A]
def resumen(X, a, b): return round(np.mean([r["coste"] for r in X if a <= r["semana"] <= b]))
for nom, X in [("nada", A), ("reentrenar", B), ("arreglar", C), ("nunca", nunca)]:
    print(nom, "60-79:", resumen(X, 60, 79), " 80-103:", resumen(X, 80, 103), " 88-103:", resumen(X, 88, 103))
print("PSI semanas 78-83:", [r["psi"] for r in A[18:24]], " avisos 76-84:", [r["avisos"] for r in A[16:25]])
json.dump(dict(nada=A, reentrenar=B, arreglar=C, nunca=nunca, factor=factor, n_nuevos=len(nuevos)), open("../produccion.json", "w"))
