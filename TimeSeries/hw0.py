from scipy import stats
import numpy as np

x_n1 = 80 / 200
x_n2 = 106 / 200


def get_d(x_n, n):
    return n ** 0.5 * (x_n - 0.5) / ((1 - 0.5) * 0.5) ** 0.5


# print(get_d(x_n1, 200))
# print(get_d(x_n2, 200))
# print(stats.norm.cdf(get_d(x_n1, 200))*200*1000)
# print((1- stats.norm.cdf(get_d(x_n2, 200)))*200*1000)
u = np.array([1, 3]).T
v = np.array([-1, 1]).T
w = np.array([0, 1]).T
x = np.array([1, 1]).T
A = np.outer(u, v)
B = np.outer(v, v)
C = np.outer(w, w)
D = np.outer(x, x)
print(A + A)
print(np.linalg.matrix_rank(A + A))
print(A + B)
print(np.linalg.matrix_rank(A + B))
print(A + C)
print(np.linalg.matrix_rank(A + C))
print(A @ B)
print(np.linalg.matrix_rank(A.dot(B)))
print(A @ C)
print(np.linalg.matrix_rank(A.dot(C)))
print(B @ D)
print(np.linalg.matrix_rank(B.dot(D)))
