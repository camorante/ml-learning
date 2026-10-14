import statistics
from fractions import Fraction
import numpy as np, pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from resumen import media, mediana, varianza, desviacion, zscore, mad, media_recortada, media_ponderada, Welford

rng = np.random.default_rng(0)
e = dict(media=0, mediana=0, var0=0, var1=0, exacta=0, std=0, z=0, mad=0, rec=0, pond=0, wel=0, fus=0)
for t in range(300):
    n = int(rng.integers(2, 400))
    x = rng.lognormal(0, 1, n)                                                   # sesgados a la derecha
    if t % 5 == 0:
        x = np.round(x, 1)                                                       # con empates
    x = (x + rng.choice([0, 10, 1000])) * 10 ** rng.uniform(-3, 3)               # desplazados y a cualquier escala
    s = np.std(x) or 1.0; a = np.abs(x).max()
    e["media"] = max(e["media"], abs(media(x) - np.mean(x)) / a)
    e["mediana"] = max(e["mediana"], abs(mediana(x) - np.median(x)) / a)
    e["var0"] = max(e["var0"], abs(varianza(x, 0) - np.var(x)) / s**2)
    e["var1"] = max(e["var1"], abs(varianza(x, 1) - pd.Series(x).var()) / s**2)
    ex = float(statistics.variance([Fraction(v) for v in x]))                                # varianza exacta (fracciones)
    e["exacta"] = max(e["exacta"], abs(varianza(x, 1) - ex) / ex)
    e["std"] = max(e["std"], abs(desviacion(x, 1) - np.std(x, ddof=1)) / s)
    z = StandardScaler().fit_transform(x[:, None]).ravel()
    e["z"] = max(e["z"], np.abs(zscore(x) - z).max(), np.abs(zscore(x, 1) - stats.zscore(x, ddof=1)).max())
    e["mad"] = max(e["mad"], abs(mad(x) - stats.median_abs_deviation(x)) / s)
    e["rec"] = max(e["rec"], abs(media_recortada(x, 0.1) - stats.trim_mean(x, 0.1)) / a)
    w = rng.uniform(0.1, 5, n)
    e["pond"] = max(e["pond"], abs(media_ponderada(x, w) - np.average(x, weights=w)) / a)
    W = Welford()
    for v in x:
        W.add(v)
    e["wel"] = max(e["wel"], abs(W.m - np.mean(x)) / a, abs(W.varianza() - ex) / ex)
    k = int(rng.integers(1, n)); A, Bw = Welford(), Welford()                               # dos trozos, fusionados
    for v in x[:k]:
        A.add(v)
    for v in x[k:]:
        Bw.add(v)
    A.fusionar(Bw)
    e["fus"] = max(e["fus"], abs(A.varianza() - ex) / ex, abs(A.m - np.mean(x)) / a)
print(f"media vs np.mean (relativo al mayor |x|)                      {e['media']:.1e}")
print(f"mediana vs np.median                                          {e['mediana']:.1e}")
print(f"varianza(ddof=0) vs np.var (relativo a la varianza)           {e['var0']:.1e}")
print(f"varianza(ddof=1) vs pandas .var()                             {e['var1']:.1e}")
print(f"varianza(ddof=1) vs exacta con fracciones (statistics)        {e['exacta']:.1e}")
print(f"desviacion vs np.std(ddof=1)                                  {e['std']:.1e}")
print(f"zscore vs StandardScaler y scipy.stats.zscore(ddof=1)         {e['z']:.1e}")
print(f"mad vs scipy.stats.median_abs_deviation                       {e['mad']:.1e}")
print(f"media_recortada vs scipy.stats.trim_mean                      {e['rec']:.1e}")
print(f"media_ponderada vs np.average(weights=)                       {e['pond']:.1e}")
print(f"Welford (un dato cada vez) vs exacta                          {e['wel']:.1e}")
print(f"Welford en dos trozos + fusionar vs exacta                    {e['fus']:.1e}")
