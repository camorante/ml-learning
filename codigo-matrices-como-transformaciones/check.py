import numpy as np
from matrices import matvec, matvec_columnas, matmul, det, solve, inversa, homogenea, rotacion

rng = np.random.default_rng(0)
e = dict(mv=0, mvc=0, mm=0, det=0, solve=0, inv=0, res=0)
for _ in range(300):
    n = int(rng.integers(2, 9)); m = int(rng.integers(2, 9)); p = int(rng.integers(2, 9))
    A = rng.normal(size=(m, n)); B = rng.normal(size=(n, p)); x = rng.normal(size=n)
    S = rng.normal(size=(n, n)); b = rng.normal(size=n)
    rel = lambda u, v: np.max(np.abs(np.array(u) - v)) / np.max(np.abs(v))
    e["mv"] = max(e["mv"], rel(matvec(A.tolist(), x.tolist()), A @ x))
    e["mvc"] = max(e["mvc"], rel(matvec_columnas(A.tolist(), x.tolist()), A @ x))
    e["mm"] = max(e["mm"], rel(matmul(A.tolist(), B.tolist()), A @ B))
    e["det"] = max(e["det"], abs(det(S.tolist()) - np.linalg.det(S)) / abs(np.linalg.det(S)))
    xs = solve(S.tolist(), b.tolist())
    e["solve"] = max(e["solve"], rel(xs, np.linalg.solve(S, b)))
    e["res"] = max(e["res"], np.max(np.abs(S @ xs - b)))
    e["inv"] = max(e["inv"], rel(inversa(S.tolist()), np.linalg.inv(S)))
print(f"matvec (por filas) vs A @ x            {e['mv']:.1e}")
print(f"matvec_columnas vs A @ x               {e['mvc']:.1e}")
print(f"matmul vs A @ B                        {e['mm']:.1e}")
print(f"det vs np.linalg.det (relativo)        {e['det']:.1e}")
print(f"solve vs np.linalg.solve (relativo)    {e['solve']:.1e}")
print(f"residuo |S x - b| máximo               {e['res']:.1e}")
print(f"inversa vs np.linalg.inv (relativo)    {e['inv']:.1e}")
print("det de una matriz singular (2.ª fila = 2 x 1.ª):", det([[1, 2, 3], [2, 4, 6], [1, 0, 1]]))
H = homogenea(rotacion(30), [5, -2]); x = [1.0, 2.0, 1.0]
R = np.array(rotacion(30)); print(f"homogénea vs R x + t                   {np.max(np.abs(np.array(matvec(H, x))[:2] - (R @ [1, 2] + [5, -2]))):.1e}")
