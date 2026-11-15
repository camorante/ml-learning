"""Junta los resultados reales en ../embed.json (sencilla) y ../tembed.json (técnica) e imprime las cifras que citan las
páginas. Antes hay que ejecutar juguete, sesgo, capacidad, parada, pesos, apagadas, inicio y modos."""
import json, os
import numpy as np
from datos import V_POOL, Y_POOL, V_TR, Y_TR, V_VA, Y_VA, REJILLA, verdad, MV, SV, MY, SY

os.chdir(os.path.dirname(os.path.abspath(__file__)))
ld = lambda f: json.load(open(f, encoding="utf-8"))
J, S, C, PA, PE, AP, IN, MO = (ld(f + ".json") for f in ("juguete", "sesgo", "capacidad", "parada", "pesos", "apagadas", "inicio", "modos"))
ruido = lambda v, y: float(np.sqrt(np.mean((verdad(v) - y) ** 2)))
print(f"suelo de ruido (la curva de verdad): entrenamiento {ruido(V_TR, Y_TR):.3f}, validación {ruido(V_VA, Y_VA):.3f} L/100 km")
print(f"escalado: velocidad (v − {MV:.2f})/{SV:.4f}; consumo (c − {MY:.2f})/{SY:.4f}")
print(f"consumo de verdad: {verdad(15):.1f} a 15 km/h, mínimo {verdad(REJILLA).min():.2f} hacia {REJILLA[verdad(REJILLA).argmin()]:.2f} km/h, {verdad(90):.1f} a 90 km/h")
d = J["0|0"]
print(f"juguete sin freno: mejor época {d['mejor']} (validación {d['vm']:.3f}, entrenamiento {d['tm']:.3f}); al final validación {d['vf']:.3f}, entrenamiento {d['tf']:.3f}")
mejor = min(J, key=lambda k: J[k]["vf"])
print(f"mejor validación AL FINAL entre las 25: {mejor} ({J[mejor]['vf']:.3f}); peor con freno: "
      f"{max((k for k in J if k != '0|0'), key=lambda k: J[k]['vf'])}")
E = dict(vp=V_POOL.tolist(), yp=Y_POOL.tolist(), vt=V_TR.tolist(), yt=Y_TR.tolist(), vv=V_VA.tolist(), yv=Y_VA.tolist(), g=REJILLA.round(2).tolist(),
         f0=np.round(verdad(REJILLA), 2).tolist(), toy=J, sesgo=S, cap=C, parada=PA, pesos=PE, apag=AP, ini=IN, modos=MO,
         suelo=round(ruido(V_VA, Y_VA), 3))
json.dump(E, open("../embed.json", "w", encoding="utf-8"), separators=(",", ":"))
json.dump(dict(ini=IN), open("../tembed.json", "w", encoding="utf-8"), separators=(",", ":"))
print("embed.json", os.path.getsize("../embed.json"), "bytes · tembed.json", os.path.getsize("../tembed.json"), "bytes")
