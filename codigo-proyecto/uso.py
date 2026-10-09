from sim import flota
from proyecto import *
d0 = limpiar(flota())
d, cols = variables(d0)
tr, ex = d[d.semana < 48], d[d.semana.between(48, 59)]
m = modelo().fit(tr[cols], tr.averia_sig)
c, fn, fp = coste(ex.averia_sig.values, m.predict_proba(ex[cols])[:, 1], T_COSTE)
print(round(c / len(ex) * 300), fn, fp)
