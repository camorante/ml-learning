import numpy as np
from scipy import stats
from sim import EUR12 as eur, KM12 as km
from ic import bootstrap, ic_media

eur_km = lambda e, k, axis=-1: np.sum(e, axis=axis) / np.sum(k, axis=axis)
print(f"media {eur.mean():.2f} €  mediana {np.median(eur):.1f} €  €/km {eur_km(eur, km):.4f}   t para la media: {np.round(ic_media(eur), 2)}")
for nombre, est in [("media", np.mean), ("mediana", np.median)]:
    for met, sm in [("percentil", "percentile"), ("basico", "basic"), ("BCa", "BCa")]:
        r = stats.bootstrap((eur,), est, n_resamples=9999, method=sm, rng=np.random.default_rng(7))
        mio = bootstrap(eur, est, B=9999, metodo=met, seed=7)
        print(f"{nombre:8s} {met:9s} scipy [{r.confidence_interval.low:7.3f}, {r.confidence_interval.high:7.3f}]"
              f"  ic.py [{mio[0]:7.3f}, {mio[1]:7.3f}]  error estándar {r.standard_error:.3f}")
for sm in ("percentile", "BCa"):          # cociente: se remuestrean viajes enteros (paired=True)
    r = stats.bootstrap((eur, km), eur_km, paired=True, n_resamples=9999, method=sm, rng=np.random.default_rng(7))
    print(f"€/km     {sm:10s} [{r.confidence_interval.low:.4f}, {r.confidence_interval.high:.4f}]  error estándar {r.standard_error:.4f}")
# método delta para el cociente R = Σe/Σk: Var(R) ≈ Σ(e_i − R k_i)² / (n (n − 1) k̄²)
R = eur.sum() / km.sum(); n = len(eur); d = eur - R * km
se = np.sqrt(np.sum(d**2) / (n * (n - 1))) / km.mean()
print(f"€/km método delta: {R:.4f} ± 1,96·{se:.4f} = [{R - 1.96 * se:.4f}, {R + 1.96 * se:.4f}]")
print("remuestreos distintos posibles con n = 12: C(23, 12) =", __import__("math").comb(23, 12))
