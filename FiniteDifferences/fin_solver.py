# fin_solver.py
import numpy as np
import math
from dataclasses import dataclass
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class FinParams:
    T_inf: float = 288.0  # K
    T_L: float = 473.0  # K
    L: float = 0.10  # m
    d: float = 1e-3  # m
    k: float = 20.0  # W/(m K)
    h: float = 50.0  # W/(m^2 K)

    @property
    def r(self) -> float:
        return self.d / 2

    @property
    def A_c(self) -> float:
        return math.pi * self.r**2

    @property
    def P(self) -> float:
        return math.pi * self.d

    @property
    def m(self) -> float:
        # fin parameter: m = sqrt(hP/(kA))
        return math.sqrt(self.h * self.P / (self.k * self.A_c))


def analytical_T(x: np.ndarray, p: FinParams) -> np.ndarray:
    """Insulated-tip fin solution."""
    m = p.m
    return (np.cosh(m * (p.L - x)) / np.cosh(m * p.L)) * (p.T_L - p.T_inf) + p.T_inf


def solve_temperature_sparse(nx: int, p: FinParams):
    """
    Solve 1D steady fin equation with Dirichlet at x=0, Neumann at x=L.
    Unknown nodes are 1..nx-1 (node 0 fixed).
    Returns: x, Tn
    """
    x = np.linspace(0.0, p.L, nx)
    dx = x[1] - x[0]

    # Source term: S(T) = -h(P/A)(T - T_inf) = S_P*T + S_u
    S_P = -p.h * (p.P / p.A_c)
    S_u = p.h * (p.P / p.A_c) * p.T_inf

    ke = p.k / dx**2

    # Unknown vector size
    N = nx - 1

    # Tridiagonal coefficients for unknowns (nodes 1..nx-1)
    lower = -ke * np.ones(N)  # subdiagonal
    diag = (2.0 * ke - S_P) * np.ones(N)  # diagonal
    upper = -ke * np.ones(N)  # superdiagonal
    b = S_u * np.ones(N)  # RHS

    # Left boundary (Dirichlet): T0 = T_L
    # Row for node 1 includes -ke*T0 -> move to RHS: b0 += ke*T_L
    b[0] += ke * p.T_L
    lower[0] = 0.0

    # Right boundary (Neumann insulated): dT/dx=0 at x=L
    # Ghost node: T_{N+1} = T_{N-1} -> last row: (-2ke)T_{N-1} + (2ke - S_P)T_N = S_u
    lower[-1] = -2.0 * ke
    upper[-1] = 0.0

    A_mat = diags([lower[1:], diag, upper[:-1]], offsets=[-1, 0, 1], format="csc")
    T_unknown = spsolve(A_mat, b)

    # Reconstruct full T including Dirichlet node
    Tn = np.empty(nx, dtype=float)
    Tn[0] = p.T_L
    Tn[1:] = T_unknown

    return x, Tn


def error_L2_continuous(
    x: np.ndarray, Tn: np.ndarray, p: FinParams, include_dirichlet: bool = True
) -> float:
    """
    continuous L^2 error approximation:
        ||e|| ~ sqrt(dx) * ||e||_2
    """
    Ta = analytical_T(x, p)
    err = Tn - Ta
    if not include_dirichlet:
        err = err[1:]
    dx = x[1] - x[0]
    return math.sqrt(dx) * np.linalg.norm(err, 2)


def order_p(E1: float, E2: float, dx1: float, dx2: float) -> float:
    """Observed order between two grids."""
    return math.log(E1 / E2) / math.log(dx1 / dx2)
