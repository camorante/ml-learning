"""Carrera de optimizadores en paisajes de dos pesos (las mismas cuentas que hacen los dibujos de las páginas,
con las convenciones de torch.optim). Se cuenta cuántos pasos tarda cada uno en quedar a menos de 0.05 del fondo."""
import numpy as np


def valle(k):
    """Valle estrecho ½(a² + k·b²): el peso b es k veces más «empinado» que el a."""
    return dict(f=lambda a, b: 0.5 * (a * a + k * b * b), g=lambda a, b: np.array([a, k * b]),
                inicio=(-4.5, 1.6), fondo=(0.0, 0.0))


CURVO = dict(f=lambda a, b: (1 - a) ** 2 + 20 * (b - a * a) ** 2,
             g=lambda a, b: np.array([-2 * (1 - a) - 80 * a * (b - a * a), 40 * (b - a * a)]),
             inicio=(-1.6, 2.4), fondo=(1.0, 1.0))                        # valle curvo (tipo Rosenbrock)


def correr(P, opt, lr, beta=0.9, pasos=400, llegar=0.05):
    th = np.array(P["inicio"], float); fondo = np.array(P["fondo"]); v = None; m = np.zeros(2); s = np.zeros(2)
    camino = [th.copy()]; llega = None
    for t in range(1, pasos + 1):
        g = P["g"](*th)
        if opt == "sgd":
            th = th - lr * g
        elif opt in ("momento", "nesterov"):
            v = g.copy() if v is None else beta * v + g
            th = th - lr * (g + beta * v if opt == "nesterov" else v)
        elif opt == "rmsprop":
            s = 0.99 * s + 0.01 * g * g
            th = th - lr * g / (np.sqrt(s) + 1e-8)
        else:                                                              # adam
            m = 0.9 * m + 0.1 * g; s = 0.999 * s + 0.001 * g * g
            th = th - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(s / (1 - 0.999 ** t)) + 1e-8)
        camino.append(th.copy())
        if not np.all(np.isfinite(th)) or np.abs(th).max() > 1e6:
            break
        if llega is None and np.linalg.norm(th - fondo) < llegar:
            llega = t
    return llega, np.array(camino)


if __name__ == "__main__":
    V = valle(100)
    print("valle ½(a² + 100·b²) desde (−4.5, 1.6), tasa 0.019 (el límite de SGD es 2/100 = 0.02)")
    for beta in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95):
        n, _ = correr(V, "momento", 0.019, beta, pasos=2000) if beta else correr(V, "sgd", 0.019, pasos=2000)
        print(f"   momento β = {beta:<4}: {n} pasos")
    print("valle cada vez más estrecho: SGD con tasa 1.5/k (por debajo de su límite 2/k) frente a Adam con tasa 0.3 fija")
    for k in (4, 25, 100, 400, 1600):
        s, _ = correr(valle(k), "sgd", 1.5 / k, pasos=10000); a, _ = correr(valle(k), "adam", 0.3, pasos=10000)
        print(f"   k = {k:5d}: SGD {s:5d} pasos   Adam {a} pasos")
    _, c = correr(V, "adam", 0.3, pasos=1); _, d = correr(V, "sgd", 0.019, pasos=1)
    print(f"   primer paso: SGD mueve (a, b) en ({d[1][0] - d[0][0]:+.3f}, {d[1][1] - d[0][1]:+.3f}),"
          f" Adam en ({c[1][0] - c[0][0]:+.3f}, {c[1][1] - c[0][1]:+.3f})")
    TASAS = {"valle": dict(sgd=0.019, momento=0.005, nesterov=0.01, rmsprop=0.1, adam=0.3),
             "curvo": dict(sgd=0.008, momento=0.008, nesterov=0.004, rmsprop=0.05, adam=0.3)}
    for nombre, P in (("valle", V), ("curvo", CURVO)):
        print(f"carrera del widget, {nombre} (momento y Nesterov con β = 0.9, 400 pasos como máximo):")
        for o, lr in TASAS[nombre].items():
            n, c = correr(P, o, lr)
            print(f"   {o:9s} tasa {lr:<6g} → {('llega en ' + str(n) + ' pasos') if n else 'no llega'}"
                  f"   (al final: f = {P['f'](*c[-1]):.1e})")
