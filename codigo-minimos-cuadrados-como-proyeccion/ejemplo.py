import numpy as np
from mcuadrados import con_intercepto, normales, sombrero, r2

carga = np.array([4.0, 12.0, 20.0]); y = np.array([26.0, 33.0, 34.0])
uno = np.ones(3)
media = (y @ uno) / (uno @ uno); e0 = y - media * uno
print(media, e0, e0 @ uno, e0 @ e0)                     # la media es la sombra de y sobre (1, 1, 1)
X = con_intercepto(carga)
print(X.T @ X, X.T @ y, np.linalg.det(X.T @ X).round(6))
b = normales(X, y); yhat = X @ b; r = y - yhat
print(b.round(10), yhat.round(10), r.round(10), (X.T @ r).round(10) + 0.0, round(r @ r, 10))
yc, hc = y - y.mean(), yhat - y.mean()
print(yc @ yc, round(hc @ hc, 10), round((yc @ hc) / np.sqrt((yc @ yc) * (hc @ hc)), 4), round(r2(y, yhat), 4))
H = sombrero(X); print(np.diag(H).round(4), H.trace().round(10), round((r @ r) / (3 - 2), 10))
