import numpy as np
from matrices import homogenea, rotacion, matmul, matvec
# girar 30° alrededor del depósito D = (3, 2) km: trasladar D al origen, girar, devolver
D = [3.0, 2.0]
T_menos = homogenea([[1, 0], [0, 1]], [-D[0], -D[1]])
R = homogenea(rotacion(30), [0, 0])
T_mas = homogenea([[1, 0], [0, 1]], D)
M = matmul(T_mas, matmul(R, T_menos))
print(np.round(M, 4))
p = [5.0, 2.0, 1.0]                       # cliente a 2 km al este del depósito
print("cliente girado:", np.round(matvec(M, p), 4))
print("el depósito no se mueve:", np.round(matvec(M, [3.0, 2.0, 1.0]), 4))
