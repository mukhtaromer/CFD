"""
1D Diffusion Equation (FTCS Finite Difference, Explicit Time Marching)
---------------------------------------------------------------------

Problem
-------
We solve the 1D diffusion equation on a 1D domain x ∈ [0, 2]:

    ∂u/∂t = ν ∂²u/∂x²

where u(x,t) is a scalar field (e.g., temperature or concentration) and ν is
the diffusion coefficient.

We use a "square wave" initial condition:
- u = 2 in the middle region (approximately x ∈ [0.5, 1.0])
- u = 1 elsewhere

Method
------
We discretize the PDE using the classic FTCS scheme:
- Space: central difference for the second derivative (∂²u/∂x²)
- Time : Forward Euler (explicit) time stepping

Update formula (interior nodes):
    u_i^{n+1} = u_i^n + r (u_{i+1}^n - 2u_i^n + u_{i-1}^n)
where:
    r = ν Δt / Δx²

Stability (Von Neumann condition) for FTCS diffusion in 1D:
    r ≤ 0.5

This is an iterative (time-stepping) explicit scheme: the solution is updated
step-by-step in time with no linear system solve.
"""

import numpy as np
from matplotlib import pyplot as plt


def print_sep(width=48, char="-"):
    """Print a separator line for cleaner console output."""
    print(char * width)


def main():

    print("Solving 1D Diffusion Equation using Finite Difference Method\n")

    # -------------------------------------------------------------------------
    # Numerical parameters
    # -------------------------------------------------------------------------
    nx = 41  # grid points in x
    dx = 2 / (nx - 1)  # grid spacing
    nt = 20  # number of time steps
    nu = 0.3  # diffusion coefficient
    cfl = 0.4  # safety factor for stability
    dt = cfl * dx**2 / nu  # time step

    # r is the key stability parameter for FTCS diffusion
    r = nu * dt / dx**2

    # -------------------------------------------------------------------------
    # Print parameters
    # -------------------------------------------------------------------------
    print_sep()
    print("Parameters")
    print_sep()
    print(f"nx={nx}, dx={dx}")
    print(f"nt={nt}, nu={nu}")
    print(f"dt={dt}")
    print(f"r = nu*dt/dx^2 = {r} (must be <= 0.5 for stability)")

    # -------------------------------------------------------------------------
    # Initial condition: square wave
    # -------------------------------------------------------------------------
    print_sep()
    print("Computing Initial Solution...")
    print_sep()

    # We initialize the u-array with ones
    u = np.ones(nx)

    # Raise u to 2 in the region x in [0.5, 1.0]
    # Convert x-locations to index range using dx
    u[int(0.5 / dx) : int(1 / dx) + 1] = 2

    # Save initial state for plotting later
    u0 = u.copy()

    print("Initial u:")
    print(u0)

    # -------------------------------------------------------------------------
    # Time-marching loop
    # -------------------------------------------------------------------------
    print_sep()
    print("Calculating Numerical Solution......")
    print_sep()

    for n in range(nt):  # advance nt time steps
        un = u.copy()  # store previous time level (u^n)

        # update interior nodes only (boundaries stay at initial values here)
        for i in range(1, nx - 1):
            u[i] = un[i] + r * (un[i + 1] - 2 * un[i] + un[i - 1])

    # -------------------------------------------------------------------------
    # Print final numerical solution
    # -------------------------------------------------------------------------
    print_sep()
    print("Final Numerical Solution:")
    print_sep()
    print(u)

    # -------------------------------------------------------------------------
    # Plot initial vs final numerical solution
    # -------------------------------------------------------------------------
    x = np.linspace(0, 2, nx)  # x-grid for plotting

    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(x, u0, label="Initial Solution", lw=2)
    plt.plot(x, u, label="Numerical Solution", lw=2)
    plt.title("1D Diffusion Equation (FTCS)")
    plt.xlabel("Grid Space")
    plt.ylabel("u")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
