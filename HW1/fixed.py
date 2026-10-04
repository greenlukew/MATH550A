import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# ============================================================
# Setup
# ============================================================
mu = 1.0
Nx, Ny = 32, 32
Lx, Ly = 1.0, 1.0
hx, hy = Lx / Nx, Ly / Ny

nU = Nx * (Ny + 1)
nV = (Nx + 1) * Ny
nP = Nx * Ny
nTotal = nU + nV + nP

u_idx = lambda i, j: (i % Nx) * (Ny + 1) + j
v_idx = lambda i, j: nU + (i % (Nx + 1)) * Ny + j
p_idx = lambda i, j: nU + nV + (i % Nx) * Ny + j

U_an = lambda x, y: np.sin(2*np.pi*y) * np.sin(2*np.pi*x)
V_an = lambda x, y: -3.5 + np.cos(2*np.pi*x) * (np.cos(2*np.pi*y) - 1.0)
P_an = lambda x, y: np.sin(2*np.pi*y) * np.cos(2*np.pi*x)

# ============================================================
# Build system
# ============================================================
A = lil_matrix((nTotal, nTotal))
b = np.zeros(nTotal)

# --- U equations ---
for i in range(Nx):
    for j in range(Ny + 1):
        row = u_idx(i, j)
        x, y = (i + 0.5) * hx, j * hy
        if j == 0 or j == Ny:              # Dirichlet
            A[row, row] = 1.0
            b[row] = U_an(x, y)
            continue
        A[row, u_idx(i-1, j)] += mu / hx**2
        A[row, u_idx(i+1, j)] += mu / hx**2
        A[row, u_idx(i, j-1)] += mu / hy**2
        A[row, u_idx(i, j+1)] += mu / hy**2
        A[row, row] += -2*mu/hx**2 - 2*mu/hy**2
        A[row, p_idx(i,   j)] += -1.0 / hx
        A[row, p_idx(i-1, j)] +=  1.0 / hx

# --- V equations ---
for i in range(Nx + 1):
    for j in range(Ny):
        row = v_idx(i, j)
        A[row, v_idx(i-1, j)] += mu / hx**2
        A[row, v_idx(i+1, j)] += mu / hx**2

        if j == 0:
            A[row, row] += -mu / hy**2
            b[row] -= mu / hy**2 * V_an(i * hx, 0.0)
        else:
            A[row, v_idx(i, j-1)] += mu / hy**2

        if j == Ny - 1:
            A[row, row] += -mu / hy**2
            b[row] -= mu / hy**2 * V_an(i * hx, Ly)
        else:
            A[row, v_idx(i, j+1)] += mu / hy**2

        A[row, row] += -2*mu/hx**2
        A[row, p_idx(i, j)]   += -1.0 / hy
        A[row, p_idx(i, j-1)] +=  1.0 / hy

# --- Continuity ---
for i in range(Nx):
    for j in range(Ny):
        row = p_idx(i, j)
        A[row, u_idx(i,   j)] += 1.0 / hx
        A[row, u_idx(i-1, j)] += -1.0 / hx
        A[row, v_idx(i,   j)] += 1.0 / hy
        A[row, v_idx(i,   j-1)] += -1.0 / hy
        b[row] = 0.0

# Pin pressure (removes nullspace)
row = p_idx(Nx-1, Ny-1)
A.rows[row] = [row]
A.data[row] = [1.0]
b[row] = P_an((Nx-0.5)*hx, (Ny-0.5)*hy)

sol = spsolve(A.tocsr(), b)

U = sol[:nU].reshape((Nx, Ny + 1))
V = sol[nU:nU + nV].reshape((Nx + 1, Ny))

# ============================================================
# Interpolate U, V to cell centers for streamline plotting
# ============================================================
Uc = 0.5 * (U[:, :-1] + U[:, 1:])     # (Nx, Ny)
Vc = 0.5 * (V[:-1, :] + V[1:, :])     # (Nx, Ny)

xc = (np.arange(Nx) + 0.5) * hx
yc = (np.arange(Ny) + 0.5) * hy
Xc, Yc = np.meshgrid(xc, yc, indexing='ij')

# ============================================================
# Analytical velocity at cell centers (for comparison)
# ============================================================
Uc_ex = U_an(Xc, Yc)
Vc_ex = V_an(Xc, Yc)

# ============================================================
# Plot streamlines
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# --- Streamlines of numerical solution ---
ax = axes[0]
strm = ax.streamplot(xc, yc, Uc.T, Vc.T,
                     density=1.5, color=np.sqrt(Uc.T**2 + Vc.T**2),
                     cmap='viridis', linewidth=1.2, arrowsize=1.2)
plt.colorbar(strm.lines, ax=ax, label='|u|')
ax.set_title('Streamlines — numerical solution')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
ax.set_aspect('equal')

# --- Streamlines of analytical solution ---
ax = axes[1]
strm = ax.streamplot(xc, yc, Uc_ex.T, Vc_ex.T,
                     density=1.5, color=np.sqrt(Uc_ex.T**2 + Vc_ex.T**2),
                     cmap='viridis', linewidth=1.2, arrowsize=1.2)
plt.colorbar(strm.lines, ax=ax, label='|u|')
ax.set_title('Streamlines — analytical solution')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
ax.set_aspect('equal')

# --- Error in velocity magnitude ---
ax = axes[2]
err = np.sqrt((Uc - Uc_ex)**2 + (Vc - Vc_ex)**2)
cf = ax.contourf(Xc, Yc, err, levels=30, cmap='hot')
plt.colorbar(cf, ax=ax, label='|u_num - u_exact|')
ax.set_title('Velocity error')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('stokes_streamlines.png', dpi=150)
plt.show()

print(f"max|U - U_ex| = {np.max(np.abs(Uc - Uc_ex)):.3e}")
print(f"max|V - V_ex| = {np.max(np.abs(Vc - Vc_ex)):.3e}")