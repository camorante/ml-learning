import numpy as np
from cor import covarianza, CovOnline

rng = np.random.default_rng(1)
t = 1.7e12 + np.sort(rng.uniform(0, 600_000, 2000))           # marcas de tiempo Unix en ms (10 minutos)
km = 512_345.0 + 2e-5 * (t - t[0]) + rng.normal(0, 0.05, 2000)  # odómetro (km): 72 km/h
ref = np.cov(t - t[0], km - km[0])[0, 1]                       # referencia: datos desplazados en float64
n = len(t)
ingenua = (np.sum(t * km) - n * t.mean() * km.mean()) / (n - 1)    # Σxy − n·x̄·ȳ
o = CovOnline()
for a, b in zip(t, km): o.add(a, b)
for nombre, v in [("Σxy − n·x̄·ȳ", ingenua), ("dos pasadas", covarianza(t, km)), ("Welford en línea", o.cov()), ("np.cov", np.cov(t, km)[0, 1])]:
    print(f"{nombre:18s} {v:16.3f}   error relativo {abs(v - ref) / abs(ref):.1e}")
t32, k32 = t.astype(np.float32), km.astype(np.float32)
d32 = ((t - t[0]).astype(np.float32), (km - km[0]).astype(np.float32))
print("float32 directo:", f"{abs(np.cov(t32, k32)[0, 1] - ref) / ref:.1e}",
      "  float32 tras restar el primer valor en float64:", f"{abs(np.cov(*d32)[0, 1] - ref) / ref:.1e}")
