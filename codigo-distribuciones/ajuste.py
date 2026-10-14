import numpy as np
from scipy import stats
from dist import ajuste_momentos, ajuste_lognormal_mv

rng = np.random.default_rng(4)
verdad = stats.lognorm(0.8, scale=12).ppf(0.95)            # cuantil 95 real: 44.7 min
q95 = {"momentos": [], "MV (logaritmos)": []}
for _ in range(4000):                                     # 4 000 muestras de 50 paradas
    x = rng.lognormal(np.log(12), 0.8, 50)
    for nombre, prm in [("momentos", ajuste_momentos(x, "lognormal")), ("MV (logaritmos)", ajuste_lognormal_mv(x))]:
        q95[nombre].append(np.exp(prm["mu"] + prm["s"] * stats.norm.ppf(0.95)))
print(f"cuantil 95 real: {verdad:.1f} min")
for nombre, v in q95.items():
    v = np.array(v)
    print(f"{nombre:16s} media {v.mean():5.1f}  sesgo {v.mean() - verdad:+5.1f}  desv. típica {v.std():4.1f}  RMSE {np.sqrt(np.mean((v - verdad) ** 2)):4.1f}")
