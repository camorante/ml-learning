from sim import registros
from tele import *
m, r = registros()
real = r.odo.iloc[-1] - r.odo.iloc[0]
print("odómetro real", round(real,1), "km; mensajes", len(m), "reenvíos", m.reenvio.sum())
print("tal como llegan", round(distancia_total(m),1))
a = quitar_duplicados(m); print("sin duplicados", len(a), round(distancia_total(a),1))
b = m.sort_values("t_equipo"); print("solo ordenar", round(distancia_total(b),1))
c = a.sort_values("t_equipo"); print("dedup+orden", round(distancia_total(c),1))
d = marcar_saltos(c); print("saltos detectados", d.salto_detectado.sum(), "de verdad", c.salto.sum(), "coinciden", (d.salto_detectado & d.salto).sum())
e = d[~d.salto_detectado]; print("todo limpio", round(distancia_total(e),1))
print("congelado", tramos_congelados(e.t_equipo.values, e.combustible.values, e.odo.values))
dr, c0, _ = estimar_deriva(e.t_equipo, e.t_servidor); print("deriva estimada s/h", round(dr,2), "desfase", round(c0,1))
print("minutos >80:", (m.vel>80).sum()*0.5, (e.vel>80).sum()*0.5)
