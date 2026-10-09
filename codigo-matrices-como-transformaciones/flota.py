import numpy as np
from matrices import rotacion
rng = np.random.default_rng(2)
np.set_printoptions(precision=3, suppress=True)

# ---- 1. Un viaje de 10 minutos a 10 Hz en ejes del camión: x adelante, y izquierda, z arriba
hz, T = 10, 600; t = np.arange(T * hz) / hz
a_lon = np.zeros_like(t)                                    # aceleración longitudinal (m/s²)
for t0, dur, a in [(5, 20, 0.9), (60, 8, -1.5), (90, 15, 0.8), (200, 4, -4.0), (215, 18, 0.7),
                   (330, 10, -1.2), (400, 22, 0.6), (520, 3, -3.2), (560, 6.2 / 1.3, -1.3)]:
    a_lon[(t >= t0) & (t < t0 + dur)] = a
v = np.clip(np.cumsum(a_lon) / hz, 0, None)                 # velocidad (m/s): empieza y acaba parado
w = 0.06 * np.sin(2 * np.pi * t / 75) * (v > 3)             # giro (rad/s) en curvas suaves
a_veh = np.c_[a_lon, v * w, np.full_like(t, 9.81)]          # el acelerómetro también mide la gravedad

# ---- 2. La caja está montada girada: guiñada 60°, cabeceo 4°, alabeo -6°
def Rz(g): return np.array([[np.cos(g), -np.sin(g), 0], [np.sin(g), np.cos(g), 0], [0, 0, 1]])
def Ry(g): return np.array([[np.cos(g), 0, np.sin(g)], [0, 1, 0], [-np.sin(g), 0, np.cos(g)]])
def Rx(g): return np.array([[1, 0, 0], [0, np.cos(g), -np.sin(g)], [0, np.sin(g), np.cos(g)]])
R_true = Rz(np.radians(60)) @ Ry(np.radians(4)) @ Rx(np.radians(-6))     # caja -> camión
a_dev = a_veh @ R_true + rng.normal(0, 0.15, (len(t), 3))               # = (R_trueᵀ a_veh)ᵀ por filas + ruido

# ---- 3. Estimar la orientación con dos datos: la gravedad y la aceleración del GPS
parado = v < 0.1                                            # según el GPS
z = a_dev[parado].mean(0); z /= np.linalg.norm(z)           # "arriba" en ejes de la caja: solo la gravedad
h = a_dev - np.outer(a_dev @ z, z)                          # parte horizontal (quitar la gravedad)
a_gps = np.gradient(v, 1 / hz) + rng.normal(0, 0.1, len(t)) # derivada de la velocidad GPS
recto = np.abs(w + rng.normal(0, 0.002, len(t))) < 0.01   # tramos rectos según el rumbo del GPS
x = h[recto].T @ a_gps[recto]                               # dirección "adelante": la que mejor sigue a a_gps
x -= (x @ z) * z; x /= np.linalg.norm(x)
y = np.cross(z, x)
R_est = np.vstack([x, y, z])                                # filas = ejes del camión vistos desde la caja
a_rot = a_dev @ R_est.T

print("R_est R_estᵀ - I, máx:", f"{np.abs(R_est @ R_est.T - np.eye(3)).max():.1e}", "  det:", round(np.linalg.det(R_est), 6))
ang = np.degrees(np.arccos((np.trace(R_est @ R_true.T) - 1) / 2))
print(f"error de orientación: {ang:.2f}°")
print("parado, ejes de la caja:", a_dev[parado].mean(0), "  ejes del camión:", a_rot[parado].mean(0), f"  ({parado.sum()} muestras)")
k = (t >= 200) & (t < 204)                                  # el frenazo fuerte
print("frenazo, ejes de la caja  :", a_dev[k].mean(0))
print("frenazo, ejes del camión  :", a_rot[k].mean(0), " real:", a_veh[k].mean(0))
print("RMSE por eje frente a la verdad (m/s²):", np.sqrt(np.mean((a_rot - a_veh)**2, 0)), "  sin corregir:", np.sqrt(np.mean((a_dev - a_veh)**2, 0)))
suave = lambda a: np.convolve(a, np.ones(hz) / hz, "same")  # media móvil de 1 s
eventos = lambda m: int(np.sum(np.diff(m.astype(int)) == 1))
for nombre, A in [("caja", a_dev), ("corregido", a_rot), ("real", a_veh)]:
    print(f"{nombre:9s}: frenazos (adelante < -2,5) {eventos(suave(A[:, 0]) < -2.5)}   giros bruscos (|lateral| > 2) {eventos(np.abs(suave(A[:, 1])) > 2)}")

# ---- 4. Columnas redundantes: km y millas en la misma regresión
n = 400
km = rng.uniform(20, 400, n); paradas = rng.integers(0, 15, n)
litros = 0.31 * km + 2.0 * paradas + rng.normal(0, 4, n)
millas = np.round(km * 0.621371, 1)                         # la misma distancia, redondeada a 0,1
X2 = np.c_[np.ones(n), km, paradas]; X3 = np.c_[X2, millas]
print(f"\ncond(X) sin millas {np.linalg.cond(X2):.1e}   con millas {np.linalg.cond(X3):.1e}")
co = []
for r in range(5):
    i = rng.integers(0, n, n)                               # remuestreo bootstrap
    b3 = np.linalg.lstsq(X3[i], litros[i], rcond=None)[0]; b2 = np.linalg.lstsq(X2[i], litros[i], rcond=None)[0]
    co.append(b3); print(f"  muestra {r}: con millas km={b3[1]:+8.3f} millas={b3[3]:+8.3f} (km + 0,621·millas = {b3[1] + 0.621371 * b3[3]:.3f})   sin millas km={b2[1]:.3f}")
xq = np.array([1, 250, 6, round(250 * 0.621371, 1)])
print("predicción para 250 km y 6 paradas en las 5 muestras:", np.round([xq @ c for c in co], 2))
