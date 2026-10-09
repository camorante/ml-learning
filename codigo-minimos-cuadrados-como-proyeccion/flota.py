import numpy as np
from sklearn.linear_model import HuberRegressor
from mcuadrados import con_intercepto, normales, mc_qr, mc_svd, ridge, r2, vif

rng = np.random.default_rng(3)
def viajes(n):
    """Consumo (L/100 km) según el peso total y la pendiente media de la ruta."""
    tara = 14.0 + rng.normal(0, 0.15, n)          # casi todos los camiones pesan lo mismo vacíos
    carga = rng.uniform(2, 24, n); peso = tara + carga
    pend = rng.uniform(-1, 3, n)                   # % medio de subida de la ruta
    cons = 18.0 + 0.5 * peso + 2.0 * pend + rng.normal(0, 1.5, n)
    return np.c_[carga, peso, pend], cons
X, y = viajes(400); Xte, yte = viajes(2000)
A = con_intercepto(X)
print("cond(X) =", f"{np.linalg.cond(A):.0f}", " cond(XᵀX) =", f"{np.linalg.cond(A.T @ A):.1e}")
print("VIF carga, peso, pendiente:", vif(X).round(0))
for nom, f in [("normales", normales), ("QR", mc_qr), ("SVD", mc_svd)]:
    b = f(A, y); rm = np.sqrt(np.mean((yte - con_intercepto(Xte) @ b) ** 2))
    print(f"{nom:9s}", b.round(3), " carga+peso =", round(b[1] + b[2], 3), " RMSE prueba", round(rm, 3))
# lo que no se sabe es el reparto entre carga y peso: 8 semanas de 50 viajes cada una
print("semana   carga    peso   suma   (mínimos cuadrados)  |  ridge λ=10: carga  peso")
for s in range(8):
    Xs, ys = viajes(50); b = mc_qr(con_intercepto(Xs), ys); b0, br = ridge(Xs, ys, 10.0)
    print(f"{s + 1:4d}   {b[1]:6.2f}  {b[2]:6.2f}  {b[1] + b[2]:5.2f}                       |  {br[0]:6.2f} {br[1]:6.2f}")
# quitar la columna redundante: estable e igual de bueno
B = con_intercepto(X[:, 1:]); b = mc_qr(B, y)
print("solo peso y pendiente:", b.round(3), " RMSE prueba", round(np.sqrt(np.mean((yte - con_intercepto(Xte[:, 1:]) @ b) ** 2)), 3))
# robustez: 4 viajes con el caudalímetro averiado (marca 25 L/100 km de más)
Xo, yo = viajes(60); yo[:4] += 25
bo = mc_qr(con_intercepto(Xo[:, 1:]), yo); hb = HuberRegressor(epsilon=1.35, alpha=0).fit(Xo[:, 1:], yo)
print("con 4 lecturas malas  MC:", bo.round(2), " Huber:", np.r_[hb.intercept_, hb.coef_].round(2), " (verdad: 18, 0.5, 2)")
