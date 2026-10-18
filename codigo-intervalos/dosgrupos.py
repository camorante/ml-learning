import numpy as np
from scipy import stats
from sim import depositos
from ic import ic_diferencia, ic_media

norte, sur = depositos()
for n in (20, 70, 200):
    a, b = norte[:n], sur[:n]
    w = stats.ttest_ind(b, a, equal_var=False).confidence_interval(0.95)
    p = stats.ttest_ind(b, a, equal_var=True).confidence_interval(0.95)
    bt = stats.bootstrap((b, a), lambda x, y, axis: x.mean(axis) - y.mean(axis), n_resamples=9999, method="percentile", rng=np.random.default_rng(1)).confidence_interval
    ma, mb = (ic_media(a)[1] - ic_media(a)[0]) / 2, (ic_media(b)[1] - ic_media(b)[0]) / 2
    print(f"n = {n:3d}: Sur − Norte {b.mean() - a.mean():+.3f}  Welch [{w.low:+.3f}, {w.high:+.3f}]  ic.py {np.round(ic_diferencia(b, a), 3)}"
          f"  t combinada [{p.low:+.3f}, {p.high:+.3f}]  bootstrap [{bt.low:+.3f}, {bt.high:+.3f}]  márgenes {ma:.2f} + {mb:.2f} = {ma + mb:.2f} frente a {(w.high - w.low) / 2:.2f}")

# diseño emparejado: los mismos 30 camiones antes y después de cambiar de neumáticos
r = np.random.default_rng(9); base = r.normal(32, 4.0, 30)
antes = np.round(base + r.normal(0, 1.0, 30), 1); despues = np.round(base - 0.8 + r.normal(0, 1.0, 30), 1)
d = antes - despues; c = np.cov(antes, despues)
print(f"emparejado: ahorro medio {d.mean():.3f}  IC {np.round(ic_media(d), 3)}  (ttest_rel: {np.round(stats.ttest_rel(antes, despues).confidence_interval(), 3)})")
print(f"  var(antes) {c[0, 0]:.2f} + var(después) {c[1, 1]:.2f} − 2·cov {c[0, 1]:.2f} = {c[0, 0] + c[1, 1] - 2 * c[0, 1]:.2f} = var(d) {d.var(ddof=1):.2f}")
print(f"  si se tratasen como grupos independientes: {np.round(ic_diferencia(antes, despues), 3)}")
