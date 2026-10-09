import numpy as np
from miss import imputar_media, rubin
y = np.array([50, 62, 56, np.nan, 15, 21, np.nan, np.nan]); g = np.array(["a"] * 4 + ["t"] * 4)
real = np.array([50, 62, 56, 58, 15, 21, 18, 24.])
print(real.mean(), np.nanmean(y))                                   # 38.0 40.8
print(round(np.nanstd(y, ddof=1), 1), round(imputar_media(y).std(ddof=1), 1))   # 21.3 16.1
gm = {k: np.nanmean(y[g == k]) for k in "at"}; yg = np.where(np.isnan(y), [gm[k] for k in g], y)
print(gm, yg.mean())                                                # {'a': 56.0, 't': 18.0} 37.0
rellenos = [[50, 15, 21], [62, 21, 21], [56, 15, 15]]               # sorteos dentro de cada grupo
q, u = [], []
for r in rellenos:
    z = y.copy(); z[np.isnan(z)] = r; q.append(z.mean()); u.append(z.var(ddof=1) / len(z))
qb, T, W, B = rubin(q, u)
print(np.round(q, 2), np.round(u, 1))                               # [36.25 38.5  36.25] [50.  53.8 57.5]
print(round(qb, 2), round(W, 1), round(B, 2), round(T, 1), round(np.sqrt(T), 2))   # 37.0 53.8 1.69 56.0 7.48
