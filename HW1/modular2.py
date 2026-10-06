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

        # Exact solution values
        true_U[center] = u(x[j], y[y_idx] + 0.5 * h)
        true_V[center] = v(x[j] + 0.5 * h, y[y_idx])
        true_P[center] = p(x[j] + 0.5 * h, y[y_idx] + 0.5 * h)

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
            # Dy
            matrices["rows"]["Dy"].extend([center])
            matrices["cols"]["Dy"].extend([below])
            matrices["vals"]["Dy"].extend([-1])
            true_H[center] = 3.5
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
            # Dy
            matrices["rows"]["Dy"].extend([center])
            matrices["cols"]["Dy"].extend([above])
            matrices["vals"]["Dy"].extend([1])
            true_H[center] = -3.5
            continue
        
        # Lv
        matrices["rows"]["Lv"].extend([center, center, center, center, center])
        matrices["cols"]["Lv"].extend([center, below, above, left, right])
        matrices["vals"]["Lv"].extend([-4.0, 1.0, 1.0, 1.0, 1.0])
        # Dy
        matrices["rows"]["Dy"].extend([center, center])
        matrices["cols"]["Dy"].extend([below, above])
        matrices["vals"]["Dy"].extend([-1, 1])
        # Lu
        matrices["rows"]["Lu"].extend([center, center, center, center, center])
        matrices["cols"]["Lu"].extend([center, below, above, left, right])
        matrices["vals"]["Lu"].extend([-4.0, 1.0, 1.0, 1.0, 1.0])
        # Gx
        matrices["rows"]["Gx"].extend([center, center])
        matrices["cols"]["Gx"].extend([left, right])
        matrices["vals"]["Gx"].extend([-0.5*h, 0.5*h])
        # Dx
        matrices["rows"]["Dx"].extend([center, center])
        matrices["cols"]["Dx"].extend([left, right])
        matrices["vals"]["Dx"].extend([-1, 1])
        # Gy
        matrices["rows"]["Gy"].extend([center, center])
        matrices["cols"]["Gy"].extend([below, above])
        matrices["vals"]["Gy"].extend([-0.5*h, 0.5*h])

        # Define forcing function values
        true_F[center] = f(x[j], y[y_idx] + 0.5 * h) * pow(h, 2)
        true_G[center] = g(x[j] + 0.5 * h, y[y_idx]) * pow(h, 2)
        # H[center] is already 0

        


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
          [Dx, Dy, Lp]], format="csr")

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
# SOLVE
b = true_F.reshape(-1, 1) + Gx.dot(true_P).reshape(-1, 1)
b2 = true_V.reshape(-1, 1) + Gy.dot(true_P).reshape(-1, 1)


S = spsolve(Lv, b2)
#S = spsolve(Lu, b)
# Reshare and remove ghost points from U, V, P
S = S.reshape(n, n, order="F")

# Reshape and remove ghost points from true_U, true_V
true_U = true_U.reshape(n, n, order="F")
true_V = true_V.reshape(n, n, order="F")

true_S = true_V



#plt.subplot(1, 2, 1)
#plt.spy(M, markersize=1)
#plt.show()
# Create a 2x1 grid of subplots
fig, axs = plt.subplots(2)

axs[0].pcolormesh(X,Y,S)
axs[0].set_title("Numerical solution")

axs[1].pcolormesh(X,Y,true_S)
axs[1].set_title("Exact solution")

plt.tight_layout()
plt.show()