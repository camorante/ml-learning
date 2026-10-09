import json, numpy as np
from sim import flota
from proyecto import limpiar
A = json.load(open("../ablacion.json")); P = json.load(open("../produccion.json"))
d = limpiar(flota()); edges = np.arange(1.0, 8.01, 0.25)
h = lambda v: np.histogram(v[~np.isnan(v)], edges)[0].tolist()
ref = d[d.semana < 60].vibracion.values
E = dict(abl=A, prod={k: P[k] for k in ["nada", "reentrenar", "arreglar", "nunca"]}, factor=P["factor"], n_nuevos=P["n_nuevos"],
         edges=edges.round(2).tolist(), href=h(ref), hsem=[h(d[d.semana == s].vibracion.values) for s in range(60, 104)])
json.dump(E, open("../embed.json", "w")); print(len(json.dumps(E)))
