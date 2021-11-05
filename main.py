import numpy as np
import timeit

a = [[5, 3, 0, 0],
     [3, 8, 1, 0], # на позиции 8 должна быть 6
     [0, 1, 4, -2],
     [0, 0, 1, -3]]
b = [8, 10, 3, -2]
n = len(a)
c = []
x = np.zeros(n)
eps = 10e-10

def p_method(matrix):
    n = len(matrix)
    x = np.zeros(n)
    g = np.zeros(n)
    a = np.zeros(n)
    b = np.zeros(n)
    d = np.zeros(n)
    p = np.zeros(n-1)
    q = np.zeros(n)
    for i in range(n):
      if i == n-1:
        g[i] = 0
      else: g[i] = matrix[i][i+1]
      b[i] = -matrix[i][i]
      d[i] = matrix[i][n]
      if i == 0:
        a[i] = 0
      else: a[i] = matrix[i][i-1]  
    p[0] = g[0]/b[0]
    q[0] = -d[0]/b[0]
    for i in range(1, n):
      if i != n-1 :
        p[i] = g[i]/(b[i]-a[i]*p[i-1])
      q[i] = (a[i]*q[i-1]-d[i])/(b[i]-a[i]*p[i-1])
    x[n-1] = q[n-1]
    for i in range(n-2, -1,-1):
        x[i] = p[i]*x[i+1]+q[i]
    for i in range(n):
      x[i] = float("%.4f" % x[i])
    return x

def kramer_method(les):
  n = len(les)
  x = np.zeros(n)
  tmp = list(zip(*les))
  b = tmp[-1]
  del tmp[-1]
  delta = np.linalg.det(tmp)
  if not delta:
      raise RuntimeError("No solution")
  result = []
  for i in range(n):
    a = tmp[:]
    a[i] = b
    result.append(np.linalg.det(a) / delta)
    x[i] = result[i]
    x[i] = float("%.4f" % x[i])
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
            x_new[i] = float("%.4f" % x_new[i])
        for i in range(1, n - 1):
            if abs(x_new[max] - x[max]) < abs(x_new[i + 1] - x[i + 1]):
                max = i + 1
        condition = abs(x_new[max] - x[max]) < eps
        x = x_new
    for i in range(n):
      x[i] = float("%.4f" % x[i])
    return x

for i in range(len(a)):
    c.append(a[i].copy())
    c[i].append(b[i])

print("Kramer method:")
start_time = timeit.default_timer()
print(kramer_method(c))
time = (timeit.default_timer() - start_time) * 1000
print("Runtime in milliseconds: %.3f" % time)

print("\nIteration method:")
start_time = timeit.default_timer()
print(iteration_method(a, b, eps))
time = (timeit.default_timer() - start_time) * 1000
print("Runtime in milliseconds: %.3f" % time)

print("\nThomas method:")
start_time = timeit.default_timer()
print(p_method(c))
time = (timeit.default_timer() - start_time) * 1000
print("Runtime in milliseconds: %.3f" % time)

print("\nNumPy:")
res = np.zeros(4)
for i in range(len(res)):
    res[i] = np.linalg.solve(a, b)[i]
    res[i] = float("%.4f" % res[i])
print(res)