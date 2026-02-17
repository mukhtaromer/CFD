# Here we solve the 2D Laplace equation for the steady-state temperature in a rectangular domain:
#     T_xx + T_yy = 0, discretized with centered finite differences.
# We build a 15×15 sparse linear system for the 3×5 interior unknowns using the 5-point stencil,
# and solve it with spsolve, reconstruct the full grid including boundary values, and plot contours/surface.



# import important libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# -----------------------------
# Geometry / grid 

#  100 ── 100 ─── 100 ─── 100 ─── 100
#          │       │       │       │
#   0 ─── T1,5 ── T2,5 ── T3,5 ─── 0
#          │       │       │
#   0 ─── T1,4 ── T2,4 ── T3,4 ─── 0
#          │       │       │
#   0 ─── T1,3 ── T2,3 ── T3,3 ─── 0
#          │       │       │
#   0 ─── T1,2 ── T2,2 ── T3,2 ─── 0
#          │       │       │
#   0 ─── T1,1 ── T2,1 ── T3,1 ─── 0
#          │       │       │
#   0 ───  0   ──  0 ──    0   ─── 0

# -----------------------------
# length of the domain in the x- and y-direction
Lx, Ly = 1.0, 1.5

# Total nodes including boundaries:
# x: 0, 0.25, 0.5, 0.75, 1.0  -> 5 nodes  -> 3 interior columns
# y: 0, 0.25, ..., 1.5        -> 7 nodes  -> 5 interior rows
Nx_total = 5
Ny_total = 7

dx = Lx / (Nx_total - 1)   # 1.0 / 4  = 0.25
dy = Ly / (Ny_total - 1)   # 1.5 / 6  = 0.25

# Interior unknown counts (matches T_{1..3, 1..5})
Nx_int = Nx_total - 2   # 3
Ny_int = Ny_total - 2   # 5
N = Nx_int * Ny_int     # 15

# Boundary conditions from the plot
T_top = 100.0
T_left = 0.0
T_right = 0.0
T_bottom = 0.0

# -----------------------------
# Indexing: map (i,j) -> k for interior
# i = 1..3, j = 1..5 
# We'll store i=1..3 and j=1..5, but in code use 0-based interior indices:
# ii = i-1 in [0..2], jj = j-1 in [0..4]
# k = jj*Nx_int + ii (row-major by y)
# -----------------------------
def k_of(ii, jj):
    return jj * Nx_int + ii

# -----------------------------
# Build A*T = b using 5-point Laplacian
# For uniform dx=dy: -4*T + neighbors = 0
# If a neighbor is a boundary node, it moves to RHS.
# -----------------------------
A = sp.lil_matrix((N, N), dtype=float)
b = np.zeros(N, dtype=float)

for jj in range(Ny_int):       # jj = 0..4 corresponds to j = 1..5
    for ii in range(Nx_int):   # ii = 0..2 corresponds to i = 1..3
        k = k_of(ii, jj)

        A[k, k] = -4.0

        # --- West neighbor (i-1)
        if ii - 1 >= 0:
            A[k, k_of(ii - 1, jj)] = 1.0
        else:
            # touches left boundary (T=0)
            b[k] -= T_left

        # --- East neighbor (i+1)
        if ii + 1 < Nx_int:
            A[k, k_of(ii + 1, jj)] = 1.0
        else:
            # touches right boundary (T=0)
            b[k] -= T_right

        # --- South neighbor (j-1)
        if jj - 1 >= 0:
            A[k, k_of(ii, jj - 1)] = 1.0
        else:
            # touches bottom boundary (T=0)
            b[k] -= T_bottom

        # --- North neighbor (j+1)
        if jj + 1 < Ny_int:
            A[k, k_of(ii, jj + 1)] = 1.0
        else:
            # touches top boundary (T=100)
            b[k] -= T_top

# Convert + solve
A = A.tocsr()
T_unknown = spla.spsolve(A, b)

# -----------------------------
# Put solution back into a full grid for plotting
# Full grid shape: (Ny_total, Nx_total)
# Boundaries filled with BCs, interior filled with solved values
# -----------------------------
T_full = np.zeros((Ny_total, Nx_total), dtype=float)

# We apply boundary conditions
#T_full[0, :] = T_bottom         # bottom y=0
T_full[-1, :] = T_top           # top y=Ly
#T_full[:, 0] = T_left           # left x=0
#T_full[:, -1] = T_right         # right x=Lx

# Fill interior 
for jj in range(Ny_int):
    for ii in range(Nx_int):
        T_full[jj + 1, ii + 1] = T_unknown[k_of(ii, jj)]

# -----------------------------
# Print interior points
# T_{i,j} with i=1..3, j=1..5
# -----------------------------
print("Interior temperatures):")
for j in range(1, Ny_int + 1):   # 1..5
    row = []
    for i in range(1, Nx_int + 1):  # 1..3
        ii, jj = i - 1, j - 1
        row.append(f"T_{{{i},{j}}}={T_unknown[k_of(ii,jj)]:8.3f}")
    print("  " + "  ".join(row))

# -----------------------------
# Plot
# -----------------------------
x = np.linspace(0, Lx, Nx_total)
y = np.linspace(0, Ly, Ny_total)
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(6, 4.5))
cf = plt.contourf(X, Y, T_full, levels=150, cmap="nipy_spectral")
plt.colorbar(cf, label="T")
# plot isolines where T is constant
#plt.contour(X, Y, T_full, levels=15, colors="white", linewidths=0.7) 

# draw grid lines
# for xv in x:
#     plt.plot([xv, xv], [0, Ly], linewidth=0.6)
# for yv in y:
#     plt.plot([0, Lx], [yv, yv], linewidth=0.6)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Laplace solution ")
plt.tight_layout()



from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, T_full)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("T")
ax.set_title("Surface plot of T(x,y)")
plt.tight_layout()
plt.show()



plt.show()
