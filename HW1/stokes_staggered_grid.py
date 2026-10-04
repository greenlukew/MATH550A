import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

n = 100
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
# for ghost points
n = n + 2
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

matrices = {}
matrices["rows"] = matrices["cols"] = matrices["vals"] = {
    "Lu": [],
    "Lv": [],
    "Gx": [],
    "Gy": [],
    "Dx": [],
    "Dy": [],
}

# Define lists for exact soolutions
true_U = np.zeros(n * n)
true_V = np.zeros(n * n)
true_P = np.zeros(n * n)

# Define lists for exact forcing functions
true_F = np.zeros(n * n)
true_G = np.zeros(n * n)
true_H = np.zeros(n * n)  # This will not change, H = 0 to enforce divergence(U,V) = 0


def populate(i, j, matrix):
    global matrices

    # Define indices
    center = j * n + i
    y_idx = n - i - 1
    below = center + 1
    above = center - 1
    left = (j - 1) * n + i
    right = (j + 1) * n + i

    if matrix in ["Lu", "Lv"]:
        matrices["rows"][matrix].extend([center, center, center, center, center])
        matrices["cols"][matrix].extend([center, below, above, left, right])
        matrices["vals"][matrix].extend([-4.0, 1.0, 1.0, 1.0, 1.0])
    elif matrix in ["Gx", "Dx"]:
        matrices["rows"][matrix].extend([center, center])
        matrices["cols"][matrix].extend([left, right])
        matrices["vals"][matrix].extend([-h, h])
    elif matrix in ["Gy", "Dy"]:
        matrices["rows"][matrix].extend([center, center])
        matrices["cols"][matrix].extend([below, above])
        matrices["vals"][matrix].extend([-h, h])
    else:
        None


# Populate
for j in range(n):
    for i in range(n):
        center = j * n + i
        y_idx = n - i - 1

        # Boundary conditions (ABOVE / BELOW)
        if i == 1 or i == n - 1:
            # Lu
            matrices["rows"]["Lu"].extend([center, center])
            matrices["cols"]["Lu"].extend([above, center])
            matrices["vals"]["Lu"].append([0.5, 0.5])
            true_F[center] = 0  # boundary condition

            # Lv
            matrices["rows"]["Lv"].append(center)
            matrices["cols"]["Lv"].append(center)
            matrices["vals"]["Lv"].append(1.0)
            true_G[center] = -3.5  # boundary condition

            populate(i, j, "Gx")
            populate(i, j, "Gy")
            populate(i, j, "Dx")
            populate(i, j, "Dy")

            continue

        populate(i, j, "Lu")
        populate(i, j, "Lv")
        populate(i, j, "Gx")
        populate(i, j, "Gy")
        populate(i, j, "Dx")
        populate(i, j, "Dy")

        # Define forcing function values
        true_F[center] = f(x[j], y[y_idx] + 0.5 * h) * pow(h, 2)
        true_G[center] = g(x[j] + 0.5 * h, y[y_idx]) * pow(h, 2)
        # H[center] is already 0

        # Exact solution values
        true_U[center] = u(x[j], y[y_idx] + 0.5 * h)
        true_V[center] = v(x[j] + 0.5 * h, y[y_idx])
        true_P[center] = p(x[j] + 0.5 * h, y[y_idx] + 0.5 * h)

l = n * n
Lu = coo_matrix((matrices["vals"]["Lu"], ((matrices["rows"]["Lu"], (matrices["cols"]["Lu"])), shape=(l, l)).tocsr()
Lv = coo_matrix((matrices["vals"]["Lv"], ((matrices["rows"]["Lv"], (matrices["cols"]["Lv"])), shape=(l, l)).tocsr()

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
