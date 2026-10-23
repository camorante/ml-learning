import time
import numpy as np
from valor import Valor, grad_inverso, grad_directo, grad_numerico, error_relativo, sigmoide

# x = (velocidad km/h, carga t, distancia km) → f(x) = (litros, horas) → g = euros
def f(v, m, d):
    litros = (20 + 0.0016 * v ** 2 + 0.4 * m) * d / 100
    return litros, d / v
def g(litros, horas): return 1.6 * litros + 30 * horas
x = np.array([80.0, 18.0, 400.0])
v, m, d = x
Jf = np.array([[0.0032 * v * d / 100, 0.4 * d / 100, (20 + 0.0016 * v ** 2 + 0.4 * m) / 100],
               [-d / v ** 2, 0.0, 1 / v]])         # filas: ∂litros/∂(v, m, d), ∂horas/∂(v, m, d)
grad_g = np.array([1.6, 30.0])                     # ∂€/∂(litros, horas)
cadena = Jf.T @ grad_g                             # ∇(g∘f) = J_fᵀ ∇g  (producto vector-jacobiano)
euros = lambda a, b, c: g(*f(a, b, c))
print("euros del viaje:", round(euros(*x), 2))
print("J_fᵀ ∇g      =", cadena.round(5), " (€ por km/h, por t, por km)")
print("modo inverso =", grad_inverso(euros, x)[1].round(5))
print("dif. centr.  =", grad_numerico(euros, x, 1e-4).round(5))

# coste: gradiente de una función de n entradas, modo directo (n pasadas) frente a inverso (1 pasada)
rng = np.random.default_rng(1)
for n in (10, 40, 160):
    A = rng.normal(size=(8, n)) / np.sqrt(n)
    def h(*xs):
        return sum(sigmoide(sum(A[k, j] * xs[j] for j in range(n))) for k in range(8))
    xs = rng.normal(size=n)
    t0 = time.perf_counter(); gi = grad_inverso(h, xs)[1]
    t1 = time.perf_counter(); gd = grad_directo(h, xs)
    t2 = time.perf_counter(); h(*[Valor(a) for a in xs])      # solo hacia delante
    t3 = time.perf_counter()
    print(f"n = {n:3d}: f con Valor {1e3 * (t3 - t2):5.2f} ms   inverso {1e3 * (t1 - t0):5.2f} ms"
          f"   directo {1e3 * (t2 - t1):6.1f} ms   iguales: {error_relativo(gi, gd):.0e}")
