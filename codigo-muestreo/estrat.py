import numpy as np
import pandas as pd
from sim import poblacion, RUTAS
from muestreo import asignar, estratificado, efecto_diseno

d, ruta, camion = poblacion()
df = pd.DataFrame({"min": d, "ruta": pd.Categorical.from_codes(ruta, RUTAS), "camion": camion})
N = len(df); sigma = df["min"].std(ddof=0)
tab = df.groupby("ruta", observed=True)["min"].agg(N_h="size", media="mean", S_h="std")
print(tab.round(2))

n = 60
tab["prop"] = asignar(n, tab.N_h)
tab["neyman"] = asignar(n, tab.N_h, tab.S_h)
print(tab[["prop", "neyman"]].T)

def muestra_estratificada(df, n_h, seed):
    """Muestra aleatoria simple dentro de cada estrato, con n_h por estrato."""
    return pd.concat([g.sample(n=int(n_h[k]), random_state=seed)
                      for k, g in df.groupby("ruta", observed=True)])

s = muestra_estratificada(df, tab["neyman"], seed=7)
m, se = estratificado(s["min"], s["ruta"].cat.codes, tab.N_h)
print(f"una muestra de Neyman: media ponderada {m:.2f} ± {se:.2f}   media sin ponderar {s['min'].mean():.2f}  (μ = {df['min'].mean():.2f})")

W = tab.N_h / N
print(f"error estándar teórico, n = 60:  simple {sigma/np.sqrt(n):.2f}"
      f"   proporcional {np.sqrt((W*tab.S_h**2).sum()/n):.2f}   Neyman {(W*tab.S_h).sum()/np.sqrt(n):.2f}")

# conglomerados: 3 camiones × 20 paradas. Correlación intraclase (ICC) por ANOVA de un factor
g = df.groupby("camion")["min"]
k, m_c = g.ngroups, 200
msb = m_c * ((g.mean() - df["min"].mean()) ** 2).sum() / (k - 1)
msw = ((df["min"] - g.transform("mean")) ** 2).sum() / (N - k)
icc = (msb - msw) / (msb + (m_c - 1) * msw)
deff = efecto_diseno(20, icc)
print(f"ICC entre paradas del mismo camión = {icc:.3f};  efecto de diseño con 20 por camión = {deff:.1f}"
      f";  error ≈ {sigma/np.sqrt(n)*np.sqrt(deff):.2f};  n efectivo ≈ {n/deff:.1f}")
