"""Genera ../embed.json (datos de la versión sencilla) e imprime los números que se citan en el texto."""
import json, numpy as np
import flota12, imagenes, huecos

def svd_firmada(X):
    """SVD con el signo de cada capa elegido para que la suma del patrón de columnas sea positiva."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    for k in range(len(s)):
        if Vt[k].sum() < 0 or (abs(Vt[k].sum()) < 1e-9 and Vt[k][np.argmax(abs(Vt[k]))] < 0):
            U[:, k] *= -1; Vt[k] *= -1
    return U, s, Vt

def angulos(A):
    """A = R(fi) · diag(s1, ±s2) · R(-teta), con teta en (-90°, 90°]."""
    U, s, Vt = np.linalg.svd(A)
    v1 = Vt[0]
    if v1[0] < 0 or (v1[0] == 0 and v1[1] < 0): v1 = -v1
    teta = np.degrees(np.arctan2(v1[1], v1[0])); u1 = A @ v1 / s[0]
    fi = np.degrees(np.arctan2(u1[1], u1[0]))
    return s, teta, fi, np.sign(np.linalg.det(A))

D = {}
for nom, A in [("ejemplo", [[2, 2], [-1, 2]]), ("inclinar", [[1, 1], [0, 1]]), ("casi plana", [[1.5, 1.3], [0.5, 0.6]]),
               ("aplastar", [[1, 0.5], [2, 1]]), ("reflejar", [[-1, 0], [0, 1]])]:
    s, t, f, sg = angulos(np.array(A, float))
    print(f"{nom:10s} σ = {s.round(4)}  1.er giro {-t:.2f}°  2.º giro {f:.2f}°  signo det {sg:+.0f}  σ1/σ2 = {s[0] / max(s[1], 1e-12):.2f}")

X, verdad, anom = flota12.tabla(seed=1)
U, s, Vt = svd_firmada(X)
e = s**2 / (s**2).sum()
print("flota 12×14  σ =", s.round(1)); print("  % por capa", (100 * e).round(2)); print("  % acumulado", np.cumsum(100 * e).round(2))
for k in range(0, 5):
    R = X - (U[:, :k] * s[:k]) @ Vt[:k]
    o = np.argsort(-np.abs(R).ravel())[:3]
    print(f"  k={k} error medio {np.abs(R).mean():.1f} km  mediana |resto| {np.median(np.abs(R)):.1f}  mayores:",
          [(int(i // 14) + 1, flota12.DIAS[i % 14] + str(i % 14 + 1), int(round(R.ravel()[i]))) for i in o])
print("  capa 1 patrón (máx=1):", (Vt[0] / Vt[0].max()).round(2), " intensidad km:", (U[:, 0] * s[0] * Vt[0].max()).round(0))
print("  capa 2 patrón:", (Vt[1] / abs(Vt[1]).max()).round(2), " intensidad:", (U[:, 1] * s[1] * abs(Vt[1]).max()).round(0))
D["F"] = dict(X=X.astype(int).tolist(), U=U.round(5).tolist(), s=s.round(3).tolist(), Vt=Vt.round(5).tolist(), dias=flota12.DIAS)

for nom, img, kk in [("camion", imagenes.camion(), 30), ("act", imagenes.actividad(), 24)]:
    Ui, si, Vti = np.linalg.svd(img, full_matrices=False)
    D[nom] = dict(U=Ui[:, :kk].round(4).tolist(), s=si[:kk].round(4).tolist(), Vt=Vti[:kk].round(4).tolist(),
                  tot=float((img**2).sum()), max=float(img.max()), m=img.shape[0], n=img.shape[1])
imagenes.info(imagenes.camion(), "camión"); imagenes.info(imagenes.actividad(), "actividad")

V, M = huecos.tabla(); t, r = huecos.capa1(V, M); c = t.mean(); t, r = t / c, r * c; P = np.outer(t, r)
h = ~M; cm = np.array([V[M[:, j], j].mean() for j in range(6)])
print("huecos: factor camión", t.round(3), " base ruta", r.round(2))
print("  error medio huecos 1 capa", np.abs(P - V)[h].mean().round(2), "máx", np.abs(P - V)[h].max().round(2),
      " | media de la ruta", np.abs(cm[None, :] - V)[h].mean().round(2), " | conocidas", np.abs(P - V)[M].mean().round(2))
D["H"] = dict(V=V.tolist(), M=M.astype(int).tolist(), P=P.round(2).tolist(), t=t.round(3).tolist(), r=r.round(2).tolist(),
              cm=cm.round(2).tolist(), rutas=huecos.RUTAS)
json.dump(D, open("../embed.json", "w"), separators=(",", ":"))
print("embed.json", len(json.dumps(D, separators=(",", ":"))))
