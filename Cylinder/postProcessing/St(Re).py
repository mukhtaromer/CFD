from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------- EDIT THESE ---------
base = Path(
    "/Users/mukhtaromer/Desktop/Openfoam/run/ov8a/postProcessing/forceCoeffsIncompressible"
)
cases = [
    # (Re, folder, tmin, tmax)
    (100, "0", 10.0, 50.0),
    (400, "50", 60.0, 100.0),
    (1000, "100", 110.0, 150.0),
    (4000, "150", 160.0, 200.0),
]
col_time = 0  # column 1 in file -> index 0
col_Cl = 4  # column 5 in file -> index 4
# ------------------------------


def load_coeff(path: Path):
    # coefficient.dat often has comment lines starting with %
    data = []
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%") or s.startswith("#"):
                continue
            parts = s.split()
            data.append([float(x) for x in parts])
    return np.array(data)


def estimate_frequency_fft(t, y):
    # Remove mean
    y = y - np.mean(y)

    # Resample to uniform dt (FFT needs uniform spacing)
    # Use median dt as target
    dt = np.median(np.diff(t))
    tu = np.arange(t[0], t[-1], dt)
    yu = np.interp(tu, t, y)

    # FFT
    n = len(yu)
    yf = np.fft.rfft(yu)
    freqs = np.fft.rfftfreq(n, d=dt)

    # ignore zero frequency peak
    mag = np.abs(yf)
    mag[0] = 0.0

    # pick dominant frequency
    k = np.argmax(mag)
    return freqs[k]


Re_list = []
St_list = []

for Re, folder, tmin, tmax in cases:
    file = base / folder / "coefficient.dat"
    arr = load_coeff(file)

    t = arr[:, col_time]
    cl = arr[:, col_Cl]

    # select window
    mask = (t >= tmin) & (t <= tmax)
    t_w = t[mask]
    cl_w = cl[mask]

    f = estimate_frequency_fft(t_w, cl_w)

    # With U=1 and D=1 -> St = f
    St = f

    Re_list.append(Re)
    St_list.append(St)

    print(f"Re={Re:>5}  window=[{tmin},{tmax}]  f={f:.5f}  St={St:.5f}")

# Plot St vs Re
plt.figure()
plt.plot(Re_list, St_list, linestyle="None", marker="o")
plt.grid(True)
plt.xlabel("Reynolds number Re")
plt.ylabel("Strouhal number St")
plt.title("St vs Re (from Cl(t))")
# plt.savefig("St_vs_Re.png", dpi=200)
plt.show()
