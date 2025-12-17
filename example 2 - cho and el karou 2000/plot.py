from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_hermite

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
d = np.load("data/run_001.npz", allow_pickle=True)
mu = d["mu"]
V = d["V"]
n_iter = len(mu)
N = len(V)

# ---------------------------------------------------------------------
# Precompute quadrature + constants once
# ---------------------------------------------------------------------
sig = 0.5
pi = np.array([0.3, 0.7])

n_gh = 20
z, p = roots_hermite(n_gh)
z = np.sqrt(2.0) * sig * z
p = p / np.sqrt(np.pi)

inv_sig2 = 1.0 / (sig * sig)

def F(x: float) -> float:
    """Scalar function used to define x1*; vectorized over Gauss–Hermite nodes."""
    t = x + (sig * sig) / x
    expo = (t * z + 0.5 * t * t) * inv_sig2
    denom = pi[1] * np.exp(expo) + pi[0]
    numer = 1.0 + (x * z) * inv_sig2
    return float(np.sum(p * numer / denom))

def x1_star(a=0.1, b=1.0, tol=1e-8, max_iter=100):
    fa = F(a)
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = F(m)
        if abs(fm) < tol or 0.5 * (b - a) < tol:
            return m
        if fa * fm <= 0:
            b = m
        else:
            a, fa = m, fm
    return 0.5 * (a + b)

x1 = x1_star()
x0 = -(sig * sig) / x1

# global y-limits across iterations
ymin = min(m.min() for m in mu)
ymax = max(m.max() for m in mu)

out_dir = Path("transport-plot")
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Transport plot (X(v) vs v) – reuse artists
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 3), dpi=256)

ax.set_ylim(ymin - 0.5, ymax + 0.5)
ax.set_xlim(V.min() - 0.5, V.max() + 0.5)
ax.set_xlabel(r"$v$")
ax.set_ylabel(r"$X(v)$")

scatter = ax.scatter(V, mu[0], label="empirical")
ax.plot(V, [x0, x1], "o", color="C1", label="theoretical")
ax.legend(loc="upper left")

Vcol = V.reshape(-1, 1)

for k, mu_k in enumerate(mu):
    scatter.set_offsets(np.hstack((Vcol, mu_k.reshape(-1, 1))))
    ax.set_title(fr"$k = {k}$")
    fig.savefig(out_dir / f"fig{k}.jpeg", transparent=False, bbox_inches="tight")

plt.close(fig)

# ---------------------------------------------------------------------
# Convergence plot – single scatter call
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5), dpi=256)

iters = np.arange(n_iter)
x_coords = np.repeat(iters, N)
y_coords = np.concatenate(mu)

ax.scatter(x_coords, y_coords, color="C0", marker=".", label="empirical measure")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$\mu_k = \pi_0 \delta_{x^{(k)}_0} + \pi_1 \delta_{x^{(k)}_1}$")
ax.set_title(r"$J(\mu,x) = C_{\mu}(x) - \phi^{\mu,\nu}(x)$")
ax.hlines(y=[x0, x1], xmin=0, xmax=n_iter - 1, ls="--", color="C1", label="MFE")
ax.legend()

fig.savefig("ex2-cho-el-karoui.jpeg", transparent=False, bbox_inches="tight")
plt.close(fig)
