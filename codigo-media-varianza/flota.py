import numpy as np
from sim import viajes
from resumen import media, mediana, media_recortada, desviacion, mad, cv, Welford

d, base = viajes()
print(len(d), "viajes,", int(d.error.sum()), "registros con la coma perdida en", d[d.error].camion.nunique(), "camiones")

# 1. ¿Qué camiones gastan más de 35 L/100 km? Lo verdadero es su consumo base.
g = d.groupby("camion").consumo
est = {"media": g.apply(media), "mediana": g.apply(mediana), "recortada 10 %": g.apply(media_recortada)}
real = base > 35
for k, v in est.items():
    a = v.values > 35
    print(f"{k:15s} marca {a.sum():3d}   aciertos {np.sum(a & real):3d}   falsas alarmas {np.sum(a & ~real):3d}")

# 2. Resumen por camión en streaming: un Welford por (camión, día) y se fusionan.
W = {}
for (c, dd), x in d.groupby(["camion", "dia"]).consumo:
    w = Welford()
    for v in x:
        w.add(v)
    W.setdefault(c, Welford()).fusionar(w)
ref = g.std()
print("Welford por días fusionado vs pandas groupby().std(): error máx.",
      f"{max(abs(W[c].desviacion() - ref[c]) / ref[c] for c in W):.1e}")

# 3. Cazar los registros erróneos dentro de cada camión: |z| > 3,5 con media/desviación o con mediana/MAD,
#    con sus 80 viajes y con solo sus 8 primeros (dos días).
for dias in (20, 2):
    s = d[d.dia < dias]; gs = s.groupby("camion").consumo
    for nombre, centro, escala in [("media y desviación", media, desviacion), ("mediana y MAD", mediana, lambda x: mad(x, 1.4826))]:
        z = gs.transform(lambda x: (x - centro(x.values)) / escala(x.values))
        m = z.abs() > 3.5
        print(f"{4 * dias:2d} viajes  {nombre:19s} marca {m.sum():3d}   errores encontrados {np.sum(m & s.error):3d} de {s.error.sum()}")

# 4. ¿Qué varía más? Coeficiente de variación en la flota (sin los errores).
ok = d[~d.error]
print(f"CV consumo {cv(ok.consumo):.3f}   CV ralentí {cv(ok.ralenti):.3f}")
