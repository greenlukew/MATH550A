import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

n = 100
x = np.linspace(0, 5, n)
y = np.linspace(0, 5, n)
h = x[1] - x[0]

pi = np.pi

s2pi = lambda X: np.sin(2.0 * pi * X)
c2pi = lambda X: np.cos(2.0 * pi * X)
# Define functions
u = lambda X, Y: s2pi(X) * s2pi(X)
v = lambda X, Y: -3.5 + c2pi(X) * (c2pi(Y) - 1.0)
p = lambda X, Y: s2pi(Y) * c2pi(X)
f = lambda X, Y: (2.0 * pi * (1.0 - 2.0 * pi)) * s2pi(Y) * s2pi(X)
g = lambda X, Y: (2.0 * pi * (1.0 - 2.0 * pi)) * c2pi(Y) * c2pi(X) + 4.0 * pow(pi, 2) * c2pi(X)

rows = []
cols = []
Lu_vals = []
Lv_vals = []

F = np.zeros(n * n)
G = np.zeros(n * n)
true_U = np.zeros(n * n)

for j in range(n):
    for i in range(n):
        center = j * n + i
        y_idx = n - i - 1

        # Boundary conditions (TOP / BOTTOM)
        if i == 0 or i == n - 1:
            rows.append(center)
            cols.append(center)
            Lu_vals.append(1.0)
            Lv_vals.append(1.0)
            F[center] = 0
            G[center] = -3.5
            continue

        below = center + 1
        above = center - 1
        left = (j - 1) * n + i
        right = (j + 1) * n + i

        row_grid = [center, center, center, center, center]
        col_grid = [center, below, above, left, right]
        laplacian_multipliers = [-4.0, 1.0, 1.0, 1.0, 1.0]

        rows.extend(row_grid)
        cols.extend(col_grid)
        Lu_vals.extend(laplacian_multipliers)
        Lv_vals.extend(laplacian_multipliers)

        F[center] = -2 * u(x[j], y[y_idx])
        true_U[center] = u(x[j], y[y_idx])

l = n * n
M = coo_matrix((vals, (rows, cols)), shape=(l, l)).tocsr()

U = spsolve(M, F) * h ** 2
U = U.reshape(n, n)
true_U = true_U.reshape(n, n)

color_2 = "#009E73"
color_inf = "#D55E00"

X, Y = np.meshgrid(x, y)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cf1 = axes[0].contourf(X, Y, U, 100)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Numeric approximation')
fig.colorbar(cf1, ax=axes[0])
axes[0].set_aspect('equal')

cf2 = axes[1].contourf(X, Y, true_U, 100)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Solution')
fig.colorbar(cf2, ax=axes[1])
axes[1].set_aspect('equal')

plt.tight_layout()
plt.show()
