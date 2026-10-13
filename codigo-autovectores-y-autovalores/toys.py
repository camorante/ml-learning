"""Números de los juguetes de la versión sencilla (para comprobar que cuadran con el ejemplo resuelto)."""
import numpy as np
from autovectores import estacionaria, tiempo_mezcla
np.set_printoptions(precision=4, suppress=True)

print("== juguete: presets")
for nom, A in [("simétrica", [[2, 1], [1, 2]]), ("no simétrica", [[2, 1], [0, 1]]), ("giro", [[0.8, -0.6], [0.6, 0.8]]),
               ("cizalla", [[1, 1], [0, 1]]), ("estirar", [[1.5, 0], [0, 0.5]]), ("dar la vuelta", [[0.5, 1.5], [1.5, 0.5]])]:
    w, V = np.linalg.eig(np.array(A, float))
    print(nom, w.round(4), [round(float(np.degrees(np.arctan2(V[1, k].real, V[0, k].real))) % 180, 1) for k in range(2)])

print("== idea 2: potencia con A = [[2,1],[1,2]] desde (1, 0)")
A = np.array([[2., 1], [1, 2]]); x = np.array([1., 0])
for k in range(6):
    print(k, x, round(float(np.degrees(np.arctan2(x[1], x[0]))), 2), "longitud", round(float(np.linalg.norm(x)), 3),
          "cociente", round(float(np.linalg.norm(A @ x) / np.linalg.norm(x)), 4))
    x = A @ x

print("== idea 3: Markov")
P2 = np.array([[0.9, 0.1], [0.6, 0.4]])
print("dos estados: π =", estacionaria(P2, "lineal"), " autovalores", np.linalg.eigvals(P2.T).round(4), " mezcla(1%)", tiempo_mezcla(P2))
d = np.array([0., 1])
for k in range(6):
    print(" día", k, d.round(4)); d = d @ P2
P3 = np.array([[0.90, 0.03, 0.07],   # en ruta → ruta, taller, parado
               [0.30, 0.60, 0.10],   # en taller →
               [0.50, 0.02, 0.48]])  # parado →
pi = estacionaria(P3, "lineal"); w = np.linalg.eigvals(P3.T)
print("tres estados: π =", pi, " de 200 camiones:", (200 * pi).round(1), " autovalores", w.round(4), " mezcla(1%)", tiempo_mezcla(P3))
for d0 in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
    d = np.array(d0, float); out = []
    for k in range(31):
        if k in (1, 2, 5, 10, 20, 30): out.append((k, (100 * d).round(1)))
        d = d @ P3
    print(d0, out)

print("== idea 5: filtro alfa-beta, A = [[1-α, 1-α], [-β, 1-β]]")
for a, b in [(0.5, 0.05), (0.5, 0.2), (1.2, 1.0), (1.5, 1.2)]:
    M = np.array([[1 - a, 1 - a], [-b, 1 - b]]); w = np.linalg.eigvals(M)
    e = np.array([20., 0]); traj = []
    for k in range(31):
        if k in (0, 1, 2, 5, 10, 20, 30): traj.append(round(float(e[0]), 2))
        e = M @ e
    print(a, b, w.round(4), "tamaño máx", round(float(np.max(np.abs(w))), 4), traj)
