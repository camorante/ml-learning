import numpy as np, pandas as pd
from tele import theil_sen, tramos_congelados
# seis mensajes en orden de llegada; posiciones en km sobre un plano local
m = pd.DataFrame(dict(id=[1, 2, 3, 4, 5, 6], t_equipo=[0, 60, 30, 30, 90, 120],
                      x=[0, 1.0, 0.5, 0.5, 5.0, 2.0], y=[0, 0, 0, 0, 3.0, 0]))
dist = lambda d: float(np.hypot(np.diff(d.x), np.diff(d.y)).sum())
print(round(dist(m), 3))                                          # 11.151 km en orden de llegada
m = m.drop_duplicates(subset=["t_equipo", "x", "y"])             # el reenvío (id 4) desaparece
m = m.sort_values("t_equipo"); print(round(dist(m), 3))          # 10.243
v = np.hypot(np.diff(m.x), np.diff(m.y)) / (np.diff(m.t_equipo) / 3600)
print(v.round(0))                                                 # [ 60.  60. 600. 509.]
m = m[~((np.r_[0, v] > 160) & (np.r_[v, 0] > 160))]; print(round(dist(m), 3))   # 2.0
# reloj: el mensaje más rápido de cada ventana (hora del equipo en horas, diferencia en s)
x = np.array([8.0, 12.0, 16.0]); y = np.array([-79.5, -239.5, -399.5])
s, c = theil_sen(x, y); print(s, c)                               # -40.0 s por hora, 240.5
te = 11 + 3 / 60 + 20 / 3600; print(round((te + (s * te + c) / 3600 - 11) * 3600, 1))  # -1.7 s respecto a las 11:00:00
# combustible: cinco lecturas iguales en 40 min con 38 km recorridos
t = np.array([0, 10, 20, 30, 40, 50]) * 60.; f = np.array([63.1, 62.4, 62.4, 62.4, 62.4, 62.4]); odo = np.array([0, 9, 18, 28, 37, 47.])
print(tramos_congelados(t, f, odo, min_min=30))                   # [(600.0, 3000.0, 40.0, 38.0)]
