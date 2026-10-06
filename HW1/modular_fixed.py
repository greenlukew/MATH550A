import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix, bmat
from scipy.sparse.linalg import spsolve

n = 10
L = n * n
len = 1
h = 1.0 * len / n
x = np.linspace(0, len, n)
y = np.linspace(0, len, n)

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
    elif matrix in ["Gx"]:
        matrices["rows"][matrix].extend([center, center])
        matrices["cols"][matrix].extend([left, right])
        matrices["vals"][matrix].extend([-0.5*h, 0.5*h])
    elif matrix in ["Dx"]:
        matrices["rows"][matrix].extend([center, center])
        matrices["cols"][matrix].extend([left, right])
        matrices["vals"][matrix].extend([-h, h])
    elif matrix in ["Gy"]:
        matrices["rows"][matrix].extend([center, center])
        matrices["cols"][matrix].extend([below, above])
        matrices["vals"][matrix].extend([-0.5*h, 0.5*h])
    elif matrix in ["Dy"]:
        matrices["rows"][matrix].extend([center, center])
        matrices["cols"][matrix].extend([below, above])
        matrices["vals"][matrix].extend([-h, h])
    else:
        None


# Populate
for j in range(n):
    for i in range(n):
        center = j * n + i
        y_idx = i
        below = center + 1
        above = center - 1
        left = ((j - 1) % n) * n + i
        right = ((j + 1) % n) * n + i
        # Boundary conditions (TOP)
        if i == 0:
            # Equation for F
            # Lu
            matrices["rows"]["Lu"].extend([center, center, center, center])
            matrices["cols"]["Lu"].extend([center, below, left, right])
            matrices["vals"]["Lu"].extend([-5.0, 1.0, 1.0, 1.0])
            # Gx
            matrices["rows"]["Gx"].extend([center, center])
            matrices["cols"]["Gx"].extend([left, right])
            matrices["vals"]["Gx"].extend([-0.5 * h, 0.5 * h])
            # = F
            true_F[center] = f(x[j], y[y_idx] + 0.5 * h) * pow(h, 2)

            # Equation for G
            # Lv
            matrices["rows"]["Lv"].extend([center])
            matrices["cols"]["Lv"].extend([center])
            matrices["vals"]["Lv"].extend([1.0])
            true_G[center] = -3.5
            continue

        if i == n-1:
            # Equation for F
            # Lu
            matrices["rows"]["Lu"].extend([center, center, center, center])
            matrices["cols"]["Lu"].extend([center, above, left, right])
            matrices["vals"]["Lu"].extend([-5.0, 1.0, 1.0, 1.0])
            # Gx
            matrices["rows"]["Gx"].extend([center, center])
            matrices["cols"]["Gx"].extend([left, right])
            matrices["vals"]["Gx"].extend([-0.5 * h, 0.5 * h])
            # = F
            true_F[center] = f(x[j], y[y_idx] + 0.5 * h) * pow(h, 2)

            # Equation for G
            # Lv
            matrices["rows"]["Lv"].extend([center])
            matrices["cols"]["Lv"].extend([center])
            matrices["vals"]["Lv"].extend([1.0])
            true_G[center] = -3.5
            continue

        if i == 1:
            # Remove v_above in Lv equation when next to top boundary and move 3.5 to the right hand side
            # Lv
            matrices["rows"]["Lv"].extend([center, center,  center, center])
            matrices["cols"]["Lv"].extend([center, below, left, right])
            matrices["vals"]["Lv"].extend([-4.0, 1.0, 1.0, 1.0])
            # Adding 3.5 as we are moving v_above from the left hand side of the eq to the right hand side
            # G
            true_G[center] = g(x[j] + 0.5 * h, y[y_idx]) * pow(h, 2) + 3.5

            # Remove v_above in Dy equation when next to top boundary and move 3.5 to the right hand side
            # Dy
            matrices["rows"]["Dy"].extend([center])
            matrices["cols"]["Dy"].extend([below])
            # Set to -1.0 because we subtract v_center from v_above in the numerator of Dy
            matrices["vals"]["Dy"].extend([-1.0])
            # Adding 3.5 to the right hand side of the eq
            true_H[center] = 3.5

            # Populate the rest of the matrices normally
            populate(i, j, "Lu")
            populate(i, j, "Gx")
            populate(i, j, "Gy")
            populate(i, j, "Dx")
            continue

        if i == n - 2:
            # Remove v_below in Lv equation when next to top boundary and move 3.5 to the right hand side
            # Lv
            matrices["rows"]["Lv"].extend([center, center,  center, center])
            matrices["cols"]["Lv"].extend([center, above, left, right])
            matrices["vals"]["Lv"].extend([-4.0, 1.0, 1.0, 1.0])
            # Adding 3.5 as we are moving v_above from the left hand side of the eq to the right hand side
            # G
            true_G[center] = g(x[j] + 0.5 * h, y[y_idx]) * pow(h, 2) + 3.5

            # Remove v_below in Dy equation when next to top boundary and move 3.5 to the right hand side
            # Dy
            matrices["rows"]["Dy"].extend([center])
            matrices["cols"]["Dy"].extend([above])
            # Set to 1.0 because we subtract v_below from v_center in the numerator of Dy
            matrices["vals"]["Dy"].extend([1.0])
            # Subtracting 3.5 from the right hand side of the eq
            true_H[center] = -3.5

            # Populate the rest of the matrices normally
            populate(i, j, "Lu")
            populate(i, j, "Gx")
            populate(i, j, "Gy")
            populate(i, j, "Dx")
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

def list2csr(my_list):
    # Step 2: Convert the list to a 2D array
    array_2d = np.array(my_list).reshape(-1, 1)  # Reshape to a column vector

    # Step 3: Convert to CSR format
    return csr_matrix(array_2d)

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

Y = np.concatenate([true_F, true_G, true_H])

M = bmat([[Lu, Z, -Gx],
          [Z, Lv, -Gy],
          [Dx, Dy, Z]], format="csr")

S = spsolve(M, Y)
print("solved")

# Meshgrid
X, Y = np.meshgrid(x, y)

# Reshare and remove ghost points from U, V, P
U = S[:L].reshape(n, n, order="F")
V = S[L:2*L].reshape(n, n, order="F")
P = S[2*L:3*L].reshape(n, n, order="F")

# Reshape true_U, true_V
true_U = true_U.reshape(n, n, order="F")
true_V = true_V.reshape(n, n, order="F")

U = true_U
V = true_V

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