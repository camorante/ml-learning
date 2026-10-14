import statistics
import numpy as np, pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

x = np.array([32, 35, 30, 33, 37, 31, 34, 32], float)     # los 8 viajes del juguete (L/100 km)
print("np.std(x)                 ", round(np.std(x), 4))                     # ÷ n
print("np.std(x, ddof=1)         ", round(np.std(x, ddof=1), 4))             # ÷ (n − 1)
print("pd.Series(x).std()        ", round(pd.Series(x).std(), 4))            # ÷ (n − 1)
print("statistics.stdev / pstdev ", round(statistics.stdev(x), 4), round(statistics.pstdev(x), 4))
print("StandardScaler().scale_   ", round(StandardScaler().fit(x[:, None]).scale_[0], 4))   # ÷ n
print("stats.zscore(x)[4]        ", round(stats.zscore(x)[4], 4))           # ÷ n
print("(37 − 33) / S             ", round(4 / np.std(x, ddof=1), 4))         # ÷ (n − 1)
