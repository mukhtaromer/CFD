# fin_error_study.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fin_solver import (
    FinParams,
    solve_temperature_sparse,
    analytical_T,
    error_L2_continuous,
    order_p,
)


def run_study(n_list=(33, 65, 129), include_dirichlet=True):
    p = FinParams()

    rows = []
    solutions = {}  # we store solutions for plotting

    for nx in n_list:
        x, Tn = solve_temperature_sparse(nx, p)
        dx = x[1] - x[0]
        E = error_L2_continuous(x, Tn, p, include_dirichlet=include_dirichlet)

        rows.append({"nx": nx, "dx": dx, "L2_error": E})
        solutions[nx] = (x, Tn)

    df = pd.DataFrame(rows)

    # Here we compute observed order p between consecutive rows
    p_list = [np.nan]
    for i in range(1, len(df)):
        p_obs = order_p(
            df.loc[i - 1, "L2_error"],
            df.loc[i, "L2_error"],
            df.loc[i - 1, "dx"],
            df.loc[i, "dx"],
        )
        p_list.append(p_obs)
    df["p"] = p_list

    # Pretty formatting for print
    df_print = df.copy()
    df_print["dx"] = df_print["dx"].map(lambda v: f"{v:.6e}")
    df_print["L2_error"] = df_print["L2_error"].map(lambda v: f"{v:.6f}")
    df_print["p"] = df_print["p"].map(lambda v: "" if np.isnan(v) else f"{v:.4f}")

    print("\nError + convergence table:")
    print(df_print.to_string(index=False))

    return p, df, solutions


def plot_temperature_comparison(p, solutions, nx_plot=33):
    x, Tn = solutions[nx_plot]
    Ta = analytical_T(x, p)

    plt.figure(figsize=(9, 5))
    plt.plot(x, Tn, "o", label=f"Numerical (nx={nx_plot})")
    plt.plot(x, Ta, linestyle="-", label="Analytical")
    plt.xlabel("x [m]")
    plt.ylabel("T [K]")
    plt.grid(True)
    plt.legend()
    plt.title("Temperature distribution: numerical vs analytical")
    plt.show()


def plot_grid_solutions(p, solutions):
    plt.figure(figsize=(9, 5))

    # optional: make legend order consistent (small -> large)
    for nx in sorted(solutions.keys()):
        x, Tn = solutions[nx]
        plt.plot(
            x,
            Tn,
            marker="o",
            linestyle="None",  # <- circles only, no line
            label=f"nx={nx}",
        )

    plt.xlabel("x [m]")
    plt.ylabel("T [K]")
    plt.grid(True)
    plt.legend()
    plt.title("Numerical solution for multiple grids (markers only)")
    plt.show()


def plot_convergence(df):
    # Important relations:
    # 1) error vs dx (log-log) + slope ~2 reference
    dx = df["dx"].to_numpy()
    E = df["L2_error"].to_numpy()

    plt.figure(figsize=(9, 5))
    plt.loglog(dx, E, marker="o", linestyle="-", label="L2 error")

    # reference O(dx^2) line anchored at first point
    ref = E[0] * (dx / dx[0]) ** 2
    plt.loglog(dx, ref, linestyle="--", label=r"reference $\mathcal{O}(\Delta x^2)$")

    plt.xlabel(r"$\Delta x$ [m]")
    plt.ylabel(r"$\|e\|_{L^2}$")
    plt.grid(True, which="both")
    plt.legend()
    plt.title("Convergence: error vs grid spacing")
    plt.show()

    # 2) observed order p vs dx (or vs refinement step)
    plt.figure(figsize=(9, 4))
    plt.plot(df["nx"], df["p"], marker="o", linestyle="-")
    plt.xlabel("nx")
    plt.ylabel("Observed order p")
    plt.grid(True)
    plt.title("Observed order of accuracy")
    plt.show()


if __name__ == "__main__":
    p = FinParams()

    # we run the study and print error and convergence table
    p, df, solutions = run_study(n_list=(33, 65, 129), include_dirichlet=True)

    # key plots
    plot_temperature_comparison(p, solutions, nx_plot=33)
    plot_grid_solutions(p, solutions)
    plot_convergence(df)
