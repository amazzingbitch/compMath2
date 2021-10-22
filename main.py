from functools import reduce
import numpy as np
from operator import mul, sub
import timeit

a = [[5, 3, 0, 0],
     [3, 6, 1, 0],
     [0, 1, 4, -2],
     [0, 0, 1, -3]]

b = [8, 10, 3, -2]
c = []
x = np.zeros(4)
eps = 10e-5

def p_method(matrix):
    n = len(matrix)
    x = np.zeros(n)
    g = [matrix[0][1], matrix[1][2], matrix[2][3], 0]
    b = [-matrix[0][0], -matrix[1][1], -matrix[2][2], -matrix[3][3]]
    a = [0, matrix[1][0], matrix[2][1], matrix[3][2]]
    d = [matrix[0][4], matrix[1][4], matrix[2][4], matrix[3][4]]
    p1 = g[0]/b[0]
    q1 = -d[0]/b[0]
    p2 = g[1]/(b[1]-a[1]*p1)
    q2 = (a[1]*q1-d[1])/(b[1]-a[1]*p1)
    p3 = g[2]/(b[2]-a[2]*p2)
    q3 = (a[2]*q2-d[2])/(b[2]-a[2]*p2)
    q4 = (a[3]*q3-d[3])/(b[3]-a[3]*p3)
    x[3] = q4
    x[2] = p3*x[3]+q3
    x[1] = p2*x[2]+q2
    x[0] = p1*x[1]+q1
    return x

def sign(indexes):
    s = sum((0, 1)[x < y] for k, y in enumerate(indexes) for x in indexes[k + 1:])
    return (s + 1) % 2 - s % 2

def column(row):
    i = 0
    n = len(row)
    while not row[i] and i < n:
        i += 1
    return i if i < n else -1

def det(matrix):
    M = matrix[:]
    n = len(M)

    indexes = [0 for _ in range(n)]

    for i in range(n):
        k = column(M[i])
        for j in (x for x in range(n) if x != i):
            M[j] = tuple(map(sub, M[j],
                             map(mul, M[i], [M[j][k] / M[i][k]] * (n + 1))))
        indexes[i] = k

    return reduce(mul, (M[i][indexes[i]] for i in range(n)), 1) * sign(indexes)


def kramer_method(les):
    n = len(les)
    x = np.zeros(n)
    tmp = list(zip(*les))
    b = tmp[-1]
    del tmp[-1]

    delta = det(tmp)
    if not delta:
        raise RuntimeError("No solution")

    result = []
    for i in range(n):
        a = tmp[:]
        a[i] = b
        result.append(det(a) / delta)
        x[i] = result[i]
        x[i] = float("%.5f" % x[i])
    return x


def iteration_method(a, b, eps):
    n = len(a)
    max = 1
    x = np.zeros(n)
    condition = False
    while not condition:
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(a[i][j] * x_new[j] for j in range(i))
            s2 = sum(a[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / a[i][i]
            x_new[i] = float("%.5f" % x_new[i])
        for i in range(1, n - 1):
            if abs(x_new[max] - x[max]) < abs(x_new[i + 1] - x[i + 1]):
                max = i + 1
        condition = abs(x_new[max] - x[max]) < eps
        x = x_new

    return x

for i in range(len(a)):
    c.append(a[i].copy())
    c[i].append(b[i])

print("Kramer method:")
start_time = timeit.default_timer()
print(kramer_method(c))
print("Runtime in milliseconds:", (timeit.default_timer() - start_time) * 1000)

print("\nIteration method:")
start_time = timeit.default_timer()
print(iteration_method(a, b, eps))
print("Runtime in milliseconds:", (timeit.default_timer() - start_time) * 1000)

print("\nThomas method:")
start_time = timeit.default_timer()
print(p_method(c))
print("Runtime in milliseconds:", (timeit.default_timer() - start_time) * 1000)

print("\nNumPy:")
res = np.zeros(4)
for i in range(len(res)):
    res[i] = np.linalg.solve(a, b)[i]
    res[i] = float("%.5f" % res[i])
print(res)