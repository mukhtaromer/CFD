#  We solve the nonlinear(semi-linear) Poisson equation on a 2D square grid:
# *    u_xx + u_yy + u^2 = -1
# with Dirichlet boundary conditions:
# *   u = 0 on all boundaries.
#
# We discretize u_xx + u_yy with 2nd-order finite differences on a uniform grid.
# Because of the nonlinear term u^2, the resulting algebraic system is nonlinear,
# so we iterate using SOR algorithm.


import numpy as np


# * We compute the optimal relaxation factor
def omega(Nx):
    # "optimal" omega estimate for a square grid
    nx = Nx - 1
    ro = np.cos(np.pi / nx)
    ro = np.clip(ro, -0.999999999999, 0.999999999999)
    return 2.0 / (1.0 + np.sqrt(1.0 - ro**2))


def SOR(Nx, Ny, h, omega, reltol, maxiter=100000):
    """
    Nonlinear Poisson: u_xx + u_yy + u^2 = -1
    with u=0 on all boundaries, solved by pointwise damped Newton updates.
    """
    u = np.zeros((Nx, Ny))  # boundary values are automatically 0

    hn2 = h * h
    relres = np.inf
    it = 0

    while relres > reltol and it < maxiter:
        it += 1
        dusum = 0.0
        usum = 0.0

        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                # Discrete residual (scaled form): neighbors - 4u + h^2(u^2+1) = 0
                # This is equivalent to (Laplace(u) + u^2 + 1 = 0) -> u_xx+u_yy+u^2=-1
                resid = (
                    u[i - 1, j]
                    + u[i + 1, j]
                    + u[i, j - 1]
                    + u[i, j + 1]
                    - 4.0 * u[i, j]
                    + hn2 * (u[i, j] ** 2 + 1.0)
                )

                # Derivative of the residual w.r.t. u_ij (Newton denominator)
                fac1 = 4.0 - 2.0 * hn2 * u[i, j]

                if abs(fac1) < 1e-14:  # safety guard against division by ~0
                    continue

                du = omega * resid / fac1  # damped Newton step
                u[i, j] += du  # update element i,j in the u-array by adding du.

                dusum += abs(du)  # accumulate total update magnitude
                usum += abs(
                    u[i, j]
                )  # accumulate solution magnitude for relative stopping

        relres = dusum / max(usum, 1e-30)  # stopping criterion: relative total change

    return u, it, relres


def print_u(u, decimals=4, zero_tol=5e-7):
    """
    Print u as a grid with aligned columns, 4 decimals
    """
    fmt = f"{{:>{decimals+3}.{decimals}f}}"  # width chosen for neat alignment (e.g. ' 0.0178')

    for row in u:
        out = []
        for val in row:
            if abs(val) < zero_tol:
                out.append(
                    "0".rjust(decimals + 3)
                )  # align zeros with the same column width
            else:
                out.append(fmt.format(val))
        print(" ".join(out))


if __name__ == "__main__":
    Nx = 9
    Ny = 9
    h = 0.125
    reltol = 1e-5
    omega = omega(Nx)

    u, it, relres = SOR(Nx, Ny, h, omega, reltol)

    print(
        f"""Number of iterations required to reach the desired tolerance = {it}.
The total change in u in the final iteration is relres = {relres:.3e} of u's total magnitude. It is below the tolerance, tol = 1e-5, so we stop.\n"""
    )
    print("The distribution of u around the rectangle:")
    print_u(u)
