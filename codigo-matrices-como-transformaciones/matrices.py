"""Matrices como transformaciones, desde cero (listas de listas, sin NumPy en los cálculos)."""


def forma(A):
    return len(A), len(A[0])


def matvec(A, x):
    """A x por filas: cada componente es el producto punto de una fila de A con x."""
    if len(A[0]) != len(x):
        raise ValueError("columnas de A != longitud de x")
    return [sum(a * b for a, b in zip(fila, x)) for fila in A]


def matvec_columnas(A, x):
    """A x por columnas: x1 * (columna 1) + x2 * (columna 2) + ..."""
    m, n = forma(A)
    y = [0.0] * m
    for j in range(n):
        for i in range(m):
            y[i] += x[j] * A[i][j]
    return y


def matmul(A, B):
    """(AB)_ij = fila i de A · columna j de B.  Coste: m*n*p multiplicaciones."""
    m, n = forma(A)
    n2, p = forma(B)
    if n != n2:
        raise ValueError("dimensiones incompatibles")
    Bt = list(zip(*B))
    return [[sum(a * b for a, b in zip(fila, col)) for col in Bt] for fila in A]


def identidad(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def lu(A):
    """Eliminación gaussiana con pivoteo parcial: P A = L U.
    Devuelve (LU compactadas en una matriz, permutación, número de intercambios)."""
    n = len(A)
    M = [list(map(float, fila)) for fila in A]
    perm = list(range(n))
    cambios = 0
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(M[i][k]))     # pivote: el mayor en valor absoluto
        if M[p][k] == 0.0:
            raise ZeroDivisionError("matriz singular")
        if p != k:
            M[k], M[p] = M[p], M[k]
            perm[k], perm[p] = perm[p], perm[k]
            cambios += 1
        for i in range(k + 1, n):
            f = M[i][k] / M[k][k]
            M[i][k] = f                                       # guardamos el multiplicador (parte L)
            for j in range(k + 1, n):
                M[i][j] -= f * M[k][j]
    return M, perm, cambios


def det(A):
    """Determinante = producto de los pivotes, con signo (-1)^intercambios. 0 si es singular."""
    try:
        M, _, c = lu(A)
    except ZeroDivisionError:
        return 0.0
    d = -1.0 if c % 2 else 1.0
    for k in range(len(A)):
        d *= M[k][k]
    return d


def solve(A, b):
    """Resuelve A x = b con LU: sustitución hacia delante (L) y hacia atrás (U). Sin calcular la inversa."""
    M, perm, _ = lu(A)
    n = len(A)
    y = [float(b[perm[i]]) for i in range(n)]
    for i in range(n):                       # L y = P b  (L con unos en la diagonal)
        y[i] -= sum(M[i][j] * y[j] for j in range(i))
    x = [0.0] * n
    for i in reversed(range(n)):             # U x = y
        x[i] = (y[i] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def inversa(A):
    """Inversa resolviendo A X = I columna a columna (una sola factorización LU)."""
    n = len(A)
    M, perm, _ = lu(A)
    cols = []
    for k in range(n):
        e = [1.0 if perm[i] == k else 0.0 for i in range(n)]
        for i in range(n):
            e[i] -= sum(M[i][j] * e[j] for j in range(i))
        x = [0.0] * n
        for i in reversed(range(n)):
            x[i] = (e[i] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
        cols.append(x)
    return [list(f) for f in zip(*cols)]


def traspuesta(A):
    return [list(c) for c in zip(*A)]


def rotacion(grados):
    import math
    t = math.radians(grados)
    return [[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]


def homogenea(A, t):
    """Matriz (n+1)x(n+1) que aplica x -> A x + t en coordenadas homogéneas."""
    n = len(A)
    return [list(A[i]) + [t[i]] for i in range(n)] + [[0.0] * n + [1.0]]
