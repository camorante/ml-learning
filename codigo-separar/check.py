import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit, GroupKFold
from splits import kfold, time_series_split, group_kfold, date_split, blocked_kfold_purged, acc_interval
rng = np.random.default_rng(0); n_ok = 0
for t in range(300):
    n = int(rng.integers(12, 400)); k = int(rng.integers(2, 8)); sh = bool(rng.integers(0, 2)); seed = int(rng.integers(0, 1000))
    a = list(kfold(n, k, sh, seed if sh else None)); b = list(KFold(k, shuffle=sh, random_state=seed if sh else None).split(np.zeros(n)))
    assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b)); n_ok += 1
    gap = int(rng.integers(0, 5)); k2 = min(k, n // 3 - 1)
    if n - gap - (n // (k2 + 1)) * k2 <= 0: gap = 0
    a = list(time_series_split(n, k2, gap=gap)); b = list(TimeSeriesSplit(k2, gap=gap).split(np.zeros(n)))
    assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b)); n_ok += 1
    g = rng.integers(0, int(rng.integers(k, 40)), n)
    if len(np.unique(g)) >= k:
        a = list(group_kfold(g, k)); b = list(GroupKFold(k).split(np.zeros(n), groups=g))
        assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b)); n_ok += 1
        for tr, te in a: assert not set(g[tr]) & set(g[te])
print("comparaciones idénticas a sklearn:", n_ok)
# propiedades de las divisiones por fecha
d = np.repeat(np.arange(100), 7)
for tr, te in date_split(d, 4, gap=3):
    assert d[tr].max() + 3 < d[te].min(), "purga"
for tr, te in blocked_kfold_purged(d, 5, gap=2):
    lo, hi = d[te].min(), d[te].max(); assert not np.any((d[tr] >= lo - 2) & (d[tr] <= hi + 2))
print("purgas OK")
# cobertura del intervalo de Wilson
cov = np.mean([(lambda lo, hi: lo <= 0.9 <= hi)(*acc_interval(rng.binomial(100, 0.9), 100)) for _ in range(20000)])
print("cobertura Wilson n=100, p=0.9:", round(cov, 3))
