import numpy as np, pandas as pd

# 1 000 viajes: 200 con lluvia (50 con alerta de frenada brusca) y 800 sin lluvia (80 con alerta)
filas = [(True, True)] * 50 + [(True, False)] * 150 + [(False, True)] * 80 + [(False, False)] * 720
df = pd.DataFrame(filas, columns=["lluvia", "alerta"]).sample(frac=1, random_state=0)

print(pd.crosstab(df.lluvia, df.alerta, margins=True, margins_name="total"))
print("-- all"); print(pd.crosstab(df.lluvia, df.alerta, normalize="all"))       # P(lluvia y alerta)
print("-- index"); print(pd.crosstab(df.lluvia, df.alerta, normalize="index"))     # P(alerta | lluvia): cada fila suma 1
print("-- columns"); print(pd.crosstab(df.lluvia, df.alerta, normalize="columns")) # P(lluvia | alerta): cada columna suma 1
print("-- groupby"); print(df.groupby("lluvia").alerta.mean())  # lo mismo que la columna True de 'index'
print(df[df.alerta].lluvia.mean(), df.alerta.mean(), (df.lluvia & df.alerta).mean())
