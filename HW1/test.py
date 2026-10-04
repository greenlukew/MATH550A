import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix, bmat
from scipy.sparse.linalg import gmres

n = 10
len = 1
h = 1.0 * len / n
x = np.linspace(-h, len, n + 2)
y = np.linspace(-h, len, n + 2)

pi = np.pi

s2pi = lambda X: np.sin(2.0 * pi * X)
c2pi = lambda X: np.cos(2.0 * pi * X)
# Define functions
u = lambda X, Y: s2pi(X) * s2pi(Y)
v = lambda X, Y: -3.5 + c2pi(X) * (c2pi(Y) - 1.0)
p = lambda X, Y: s2pi(Y) * c2pi(X)
f = lambda X, Y: (2.0 * pi * (1.0 - 4.0 * pi)) * s2pi(Y) * s2pi(X)
g = lambda X, Y: (2.0 * pi * (-1.0 - 4.0 * pi)) * c2pi(Y) * c2pi(X) + 4.0 * pow(pi, 2) * c2pi(X)

m = ["Lu", "Lv", "Gx", "Gy", "Dx", "Dy"]
matrices = {
    "rows": {k: [] for k in m},
    "cols": {k: [] for k in m},
    "vals": {k: [] for k in m},
}
n = n + 2
L = n * n
# Define lists for exact soolutions
true_U = np.zeros(L)
true_V = np.zeros(L)
true_P = np.zeros(L)

# Define lists for exact forcing functions
true_F = np.zeros(L)
true_G = np.zeros(L)
true_H = np.zeros(L)  # This will not change, H = 0 to enforce divergence(U,V) = 0


def populate(i, j, matrix):
    global matrices

    # Define indices
    center = j * n + i
    y_idx = i #n - 1 - i
    below = center + 1
    above = center - 1
    left = ((j - 1) % n) * n + i
    right = ((j + 1) % n) * n + i
    if matrix in ["Lu", "Lv"]:
        matrices["rows"][matrix].extend([center, center, center, center, center])
        matrices["cols"][matrix].extend([center, below, above, left, right])
        matrices["vals"][matrix].extend([-4.0, 1.0, 1.0, 1.0, 1.0])
    elif matrix in ["Gx", "Dx"]:
        matrices["rows"][matrix].extend([center, center])
        matrices["cols"][matrix].extend([left, right])
        matrices["vals"][matrix].extend([-1/h, 1/h])
    elif matrix in ["Gy", "Dy"]:
        matrices["rows"][matrix].extend([center, center])
        matrices["cols"][matrix].extend([below, above])
        matrices["vals"][matrix].extend([-1/h, 1/h])
    else:
        None


def populate_val(i, j, matrix, val):
    global matrices
    center = j * n + i
    matrices["rows"][matrix].append(center)
    matrices["cols"][matrix].append(center)
    matrices["vals"][matrix].append(val)


# Populate
for j in range(n):
    for i in range(n):
        center = (j * n) + i
        y_idx = i #n - 1 - i
        # Boundary conditions (ABOVE / BELOW)
        if i == 0 or i == n - 1:
            if i == 0:
                neighbor = center + 1
            else:
                neighbor = center - 1
            # Lu
            matrices["rows"]["Lu"].extend([center, center])
            matrices["cols"]["Lu"].extend([neighbor, center])
            matrices["vals"]["Lu"].extend([0.5, 0.5])
            true_F[center] = 0  # boundary condition

            # Lv
            matrices["rows"]["Lv"].append(center)
            matrices["cols"]["Lv"].append(center)
            matrices["vals"]["Lv"].append(1.0)
            true_G[center] = -3.5  # boundary condition

            populate(i, j, "Gx")
            populate(i, j, "Dx")

            # populate_val(i, j, "Gx", 0.0)
            # populate_val(i, j, "Dx", 0.0)

            populate_val(i, j, "Gy", 0.0)
            populate_val(i, j, "Dy", 0.0)
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


def convert_to_csr(matrix):
    return coo_matrix(
        (matrices["vals"][matrix],
         (matrices["rows"][matrix], matrices["cols"][matrix])),
        shape=(L, L),
    ).tocsr()


Lu = convert_to_csr("Lu")
Lv = convert_to_csr("Lv")
Gx = convert_to_csr("Gx")
Gy = convert_to_csr("Gy")
#Dx = Gx.T.tocsr()
#Dy = Gy.T.tocsr()
Dx = convert_to_csr("Dx")
Dy = convert_to_csr("Dy")
# Zero matrix
Z = csr_matrix((L, L))

p_pin = 0
Lp = csr_matrix(([1.0], ([p_pin], [p_pin])), shape=(L, L))


M = bmat([[Lu, Z, -Gx],
          [Z, Lv, -Gy],
          [Dx, Dy, Lp]], format="csr")

row_norms = np.asarray(np.abs(M).sum(axis=1)).ravel()
zero_rows = np.where(row_norms == 0)[0]
print(zero_rows)

plt.spy(M, markersize=1)
plt.show()

Y = np.concatenate([true_F, true_G, true_H])

S = gmres(M, Y)[0]
print("solved")

# Remove ghost points from the grid
x_int = x[1:-1]                          # 5 points: 0, h, 2h, 3h, 4h
y_int = y[1:-1]

# Meshgrid
X, Y = np.meshgrid(x_int, y_int)

print(X)
print(Y)

# Reshare and remove ghost points from U, V, P
U = S[:L].reshape(n, n, order="F")[1:-1, 1:-1]
V = S[L:2*L].reshape(n, n, order="F")[1:-1, 1:-1]
P = S[2*L:3*L].reshape(n, n, order="F")[1:-1, 1:-1]

print(true_U)
print(true_V)
# Reshape and remove ghost points from true_U, true_V
true_U = true_U.reshape(n, n, order="F")[1:-1, 1:-1]
true_V = true_V.reshape(n, n, order="F")[1:-1, 1:-1]

print("true_U_grid[1,1] =", true_U[1, 1])   # expect 0
print("true_U_grid[1,2] =", true_U[1, 2])   # expect 0.559
print("true_U_grid[2,2] =", true_U[2, 2])   # expect 0.904

#U = true_U
#V = true_V

speed = np.sqrt(U**2 + V**2)

fig, ax = plt.subplots(figsize=(7, 7))

# Background: filled contours of speed
cf = ax.contourf(X, Y, speed, levels=50, cmap="viridis")

# Streamlines in white on top
ax.streamplot(X, Y, U, V, color="white", density=1.3, linewidth=1.0, arrowsize=1.2)

fig.colorbar(cf, ax=ax, label="speed")
ax.set_aspect("equal")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Streamlines (white) over speed field")
plt.show()