import numpy as np
from monitor import ks
fn, fp = 16 - 12, 110 - 12
print(fn * 4000 + fp * 150, 16 * 4000)                                    # 30700 64000
p = 110 / 300; sd = np.sqrt(300 * p * (1 - p)); print(round(sd, 2), round((65 - 110) / sd, 2))   # 8.35 -5.39
e = np.array([.25, .25, .25, .25]); a = np.array([.45, .30, .15, .10])
t = (a - e) * np.log(a / e); print(t.round(4), round(t.sum(), 4))        # [0.1176 0.0091 0.0511 0.1374] 0.3152
ref = [3.2, 3.6, 4.1, 4.5, 5.0]; act = [2.3, 2.6, 2.9, 3.4, 3.7]
from scipy import stats
print(ks(ref, act), round(stats.ks_2samp(ref, act).pvalue, 3))            # 0.6 0.357 (exacto)
print(round(2.87 / 4.10, 3), round(2.87 / 0.70, 2))                         # 0.7 4.1
