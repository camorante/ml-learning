"""Datos del widget de §7 (tembed.json): cuantiles e histogramas de la flota simulada."""
import json
import numpy as np
from scipy import stats
from sim import flota

consumo, tipo, alertas, tasa, paradas, tarde = flota()
out = {}
for key, x, lo, hi, w in [("larga", consumo[tipo == 0], 22, 42, 0.5), ("flota", consumo, 16, 42, 0.5),
                          ("paradas", paradas, 0, 120, 2.5), ("logp", np.log(paradas), 0, 5.5, 0.125)]:
    q = np.quantile(x, (np.arange(200) + 0.5) / 200)
    h, _ = np.histogram(x, bins=np.arange(lo, hi + w / 2, w))
    out[key] = dict(q=[round(float(v), 3) for v in q], m=round(float(x.mean()), 4), s=round(float(x.std(ddof=1)), 4),
                    n=int(len(x)), lo=lo, hi=hi, w=w, h=h.tolist(), fuera=int(np.sum(x >= hi)))
sl, _, esc = stats.lognorm.fit(paradas, floc=0)
out["paradas"]["ln"] = [round(float(np.log(esc)), 4), round(float(sl), 4)]
a = alertas.ravel()
out["alertas"] = dict(obs=np.bincount(a, minlength=16)[:16].tolist() + [int(np.sum(a >= 16))], n=int(a.size),
                      m=round(float(a.mean()), 4), v=round(float(a.var(ddof=1)), 4))
json.dump(out, open("../tembed.json", "w"), separators=(",", ":"))
print({k: (v.get("m"), v.get("s", v.get("v"))) for k, v in out.items()}, out["alertas"]["obs"])
