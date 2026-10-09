import time, numpy as np
rng = np.random.default_rng(1)
n = 2000
A, B = rng.normal(size=(n, n)), rng.normal(size=(n, n)); x = rng.normal(size=n)
print("asociatividad |(AB)x - A(Bx)| / |A(Bx)| =", f"{np.linalg.norm((A @ B) @ x - A @ (B @ x)) / np.linalg.norm(A @ (B @ x)):.1e}")
print("conmutatividad |AB - BA| / |AB| =", round(np.linalg.norm(A @ B - B @ A) / np.linalg.norm(A @ B), 3))
def t(f, k=3):
    best = 1e9
    for _ in range(k):
        t0 = time.perf_counter(); f(); best = min(best, time.perf_counter() - t0)
    return best
print(f"multiplicaciones: (AB)x = n^3 + n^2 = {n**3 + n**2:.2e}   A(Bx) = 2 n^2 = {2 * n**2:.2e}")
print(f"tiempo (AB)x {t(lambda: (A @ B) @ x) * 1e3:.1f} ms   A(Bx) {t(lambda: A @ (B @ x)) * 1e3:.2f} ms")
