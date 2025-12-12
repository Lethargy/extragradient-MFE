import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
from scipy.special import roots_hermite

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
opt_mu = [np.load(f"data/mu{i}.npy") for i in range(50)]
V = np.load("data/V.npy")
n_iter = len(opt_mu)
N = len(V)

# ---------------------------------------------------------------------
# Transport plot (X(v) vs v)
# ---------------------------------------------------------------------

sig = 0.5
pi = np.array([0.3, 0.7])

def F(x):
    z,p = roots_hermite(20)
    z = np.sqrt(2) * sig * z
    p = p / np.sqrt(np.pi)
    I = 0.0
    eps = np.exp(sig**(-2))

    for i in range(20):
        numerator = 1 + x*z[i]/sig**2
        denominator = pi[1] * pow(eps, (x + sig**2/x)*z[i] + 0.5*(x+sig**2/x)**2) + pi[0]
        I += p[i] * numerator / denominator

    return I

def x1_star(F, a=0.1, b=1.0, tol=1e-8, max_iter=100):
    fa = F(a)
    fb = F(b)
    
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = F(m)

        # stop if function value or interval size is small enough
        if abs(fm) < tol or 0.5 * (b - a) < tol:
            return m

        # keep the subinterval where the sign changes
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    # fallback if we hit max_iter
    return 0.5 * (a + b)

x1 = x1_star(F)
x0 = -sig**2 / x1

a = min(mu.min() for mu in opt_mu)
b = max(mu.max() for mu in opt_mu)

os.makedirs("transport-plot", exist_ok=True)

fig, ax = plt.subplots(figsize=(5, 3), dpi=256)

# Set static axis properties once
ax.set_ylim(a - 0.5, b + 0.5)
ax.set_xlim(V.min() - 0.5, V.max() + 0.5)
ax.set_xlabel(r"$v$")
ax.set_ylabel(r"$X(v)$")

# Create the artists once and reuse them
scatter = ax.scatter(V, opt_mu[0], label="empirical")
exact, = ax.plot(
    V,
    [x0,x1],
    'o',
    color="C1",
    label="theoretical",
)
ax.legend(loc="upper left")

for k in range(n_iter):
    mu_k = opt_mu[k]

    # Update scatter data
    scatter.set_offsets(np.column_stack((V, mu_k)))

    # Update title only
    ax.set_title(fr"$k = {k}$")

    fig.savefig(
        f"transport-plot/fig{k}.jpeg",
        transparent=False,
        bbox_inches="tight",
    )