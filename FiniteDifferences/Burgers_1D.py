"""
1D Viscous Burgers' Equation (Finite Difference, Explicit Time Marching)
-----------------------------------------------------------------------

Problem
-------
We solve the 1D viscous Burgers' equation on a periodic domain x ∈ [0, 2π]:

    ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²

where u(x,t) is the velocity-like field and ν is the kinematic viscosity.

To have a known reference solution, we build the initial condition from the
closed-form (analytical) solution obtained via the Hopf–Cole transformation.
Then we compare the final numerical solution against the analytical solution.

Method
------
- Spatial grid: uniform finite differences.
- Convection term u ∂u/∂x: backward difference (upwind-like).
- Diffusion term ν ∂²u/∂x²: central difference.
- Time integration: explicit time marching (Forward-Euler form).
- Boundary condition: periodic (u(0) = u(2π)).

This is an *iterative* (time-stepping) explicit scheme: u^{n+1} is computed
from u^n without solving a linear system.
"""

import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
from matplotlib import pyplot as plt


def build_analytical_u_function():
    """
    Here we construca a NumPy-callable function u(t, x, nu) for the analytical solution.

    We define:
        phi(x,t) = exp(-(x-4t)^2/(4 nu (t+1))) + exp(-(x-4t-2π)^2/(4 nu (t+1)))
    and then:
        u = -2 nu (phi_x / phi) + 4
    """
    # We  define symbolic variables
    x_sym, nu_sym, t_sym = sp.symbols("x nu t")

    # Define phi(x,t) (a sum of Gaussians shifted by 2π to help with periodicity)
    phi = sp.exp(-((x_sym - 4 * t_sym) ** 2) / (4 * nu_sym * (t_sym + 1))) + sp.exp(
        -((x_sym - 4 * t_sym - 2 * sp.pi) ** 2) / (4 * nu_sym * (t_sym + 1))
    )

    # The Derivative of phi with respect to x
    phi_x = sp.diff(phi, x_sym)

    # Analytical u expression
    u_expr = -2 * nu_sym * (phi_x / phi) + 4

    # We convert symbolic expression to a fast NumPy function
    ufunc = lambdify((t_sym, x_sym, nu_sym), u_expr, "numpy")

    return ufunc


def main():
    # Pretty-printing helps us to print results in a nice format.
    sp.init_printing(use_latex=True)

    SEPARATOR = "-" * 60
    print(SEPARATOR)
    print("Solving 1D Burgers' equation with an explicit finite-difference scheme")
    print("Convection: backward difference | Diffusion: central difference")
    print("Boundary: periodic")
    print(SEPARATOR)

    # -------------------------------------------------------------------------
    # Build analytical solution function u(t, x, nu)
    # -------------------------------------------------------------------------
    ufunc = build_analytical_u_function()

    # -------------------------------------------------------------------------
    # Numerical parameters (match the notebook)
    # -------------------------------------------------------------------------
    nx = 101  # number of spatial grid points
    nt = 20  # number of time steps
    nu = 0.07  # viscosity
    dx = 2 * np.pi / (nx - 1)  # spatial step size
    dt = dx * nu  # time step (as in the notebook; not necessarily optimal)

    # -------------------------------------------------------------------------
    # Create grid and initial condition
    # -------------------------------------------------------------------------
    x = np.linspace(0, 2 * np.pi, nx)  # spatial grid
    t0 = 0.0  # initial time

    # Evaluate analytical expression at t=0 to get initial condition
    u = np.asarray([ufunc(t0, xi, nu) for xi in x], dtype=float)

    print(SEPARATOR)
    print("Initial condition u(x,0):")
    print(SEPARATOR)
    print(u)

    # -------------------------------------------------------------------------
    # Plot initial condition
    # -------------------------------------------------------------------------
    plt.figure(figsize=(11, 7), dpi=100)
    plt.plot(x, u, marker="o", lw=2, label="Initial Solution")
    plt.title("1D Burgers' Equation (Initial Condition)")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.xlim([0, 2 * np.pi])
    plt.ylim([0, 10])
    plt.legend()
    plt.show()

    # -------------------------------------------------------------------------
    # Analytical solution at final time (for comparison)
    # -------------------------------------------------------------------------
    t_final = nt * dt
    u_analytical = np.asarray([ufunc(t_final, xi, nu) for xi in x], dtype=float)

    print(SEPARATOR)
    print(f"Analytical solution u(x,t) at t = {t_final:.6f}:")
    print(SEPARATOR)
    print(u_analytical)

    # -------------------------------------------------------------------------
    # Explicit time marching to compute numerical solution
    # -------------------------------------------------------------------------
    print(SEPARATOR)
    print("Computing numerical solution by time marching...")
    print(SEPARATOR)

    # Temporary array to store previous time step
    un = np.empty(nx)

    for n in range(nt):  # we loop over time steps
        un = u.copy()  # store u^n so we can compute u^{n+1}

        # Update interior points using:
        # u_i^{n+1} = u_i^n
        #            - u_i^n * (dt/dx) * (u_i^n - u_{i-1}^n)          [backward difference convection]
        #            + nu * (dt/dx^2) * (u_{i+1}^n - 2u_i^n + u_{i-1}^n) [central difference diffusion]
        for i in range(1, nx - 1):
            convection = un[i] * (dt / dx) * (un[i] - un[i - 1])
            diffusion = nu * (dt / dx**2) * (un[i + 1] - 2 * un[i] + un[i - 1])
            u[i] = un[i] - convection + diffusion

        # Periodic boundary condition:
        # left boundary i=0 uses neighbor at i=-2 (since i=-1 is the duplicate endpoint)
        convection_0 = un[0] * (dt / dx) * (un[0] - un[-2])
        diffusion_0 = nu * (dt / dx**2) * (un[1] - 2 * un[0] + un[-2])
        u[0] = un[0] - convection_0 + diffusion_0

        # We  enforce periodic wrap: last point equals first point
        u[-1] = u[0]

    print(SEPARATOR)
    print("Numerical solution u(x,t_final):")
    print(SEPARATOR)
    print(u)

    # -------------------------------------------------------------------------
    # Plot numerical vs analytical
    # -------------------------------------------------------------------------
    plt.figure(figsize=(11, 7), dpi=100)
    plt.plot(x, u, marker="o", lw=2, label="Numerical (FD) Solution")
    plt.plot(x, u_analytical, lw=2, label="Analytical Solution")
    plt.title("1D Burgers' Equation: Numerical vs Analytical")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.xlim([0, 2 * np.pi])
    plt.ylim([0, 10])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
