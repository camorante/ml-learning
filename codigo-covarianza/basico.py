import numpy as np, pandas as pd
from sim import CARGA8 as x, CONSUMO8 as y        # 8 camiones: toneladas y L/100 km

dx, dy = x - x.mean(), y - y.mean()
c = np.sum(dx * dy) / (len(x) - 1)
r = c / (x.std(ddof=1) * y.std(ddof=1))
print("cov", round(c, 4), " r", round(r, 4), " cov en kg", round(np.cov(x * 1000, y)[0, 1], 1), " r en kg", round(np.corrcoef(x * 1000, y)[0, 1], 4))
zx, zy = dx / x.std(ddof=1), dy / y.std(ddof=1)
print("media de zx·zy (n − 1)", round(np.sum(zx * zy) / 7, 4), " coseno(dx, dy)", round(dx @ dy / np.linalg.norm(dx) / np.linalg.norm(dy), 4))
b = c / x.var(ddof=1)
print("pendiente", round(b, 4), " = r·sy/sx", round(r * y.std(ddof=1) / x.std(ddof=1), 4), " r²", round(r**2, 4), " polyfit", np.polyfit(x, y, 1).round(4))
# las convenciones de ddof de cada librería
print("np.cov", round(np.cov(x, y)[0, 1], 4), " np.cov(bias=True)", round(np.cov(x, y, bias=True)[0, 1], 4),
      " pandas", round(pd.Series(x).cov(pd.Series(y)), 4), " np.var", round(np.var(x), 4), " pd.var", round(pd.Series(x).var(), 4))
