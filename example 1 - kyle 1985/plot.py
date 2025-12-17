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
data = np.load("data.npz", allow_pickle=True)
opt_mu = data['mu']
V = data['V']
n_iter = data['N_iter']
N = data['N_samp']

# ---------------------------------------------------------------------
# Transport plot (X(v) vs v)
# ---------------------------------------------------------------------

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
exact_line, = ax.plot(
    V,
    0.5 * V,
    "--",
    color="C1",
    label=r"$\frac{\sigma_z}{\sigma_v}(v-p_0)$ (exact)",
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

# ---------------------------------------------------------------------
# Potential plot
# ---------------------------------------------------------------------
os.makedirs("potential-plot", exist_ok=True)

potentials = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, potentials=True)
sig = 0.5

# Gauss–Hermite nodes/weights for N(0, σ²)
n_gh = 20
z, p = roots_hermite(n_gh)
z = np.sqrt(2.0) * sig * z
p = p / np.sqrt(np.pi)

def P_vec(x, mu):
    """
    Vectorized version of your P(x) for a given measure mu.

    x  : scalar or 1D array of evaluation points
    mu : 1D array of support points (same mu used inside the integral)
    """
    x_arr = np.atleast_1d(x).astype(float)      # (M,)
    X = x_arr[:, None, None]                    # (M, 1, 1)
    Z = z[None, :, None]                        # (1, n_gh, 1)
    MU = mu[None, None, :]                      # (1, 1, N)

    # unnormalised weights in i for each (x_j, z_k)
    w_unnorm = np.exp(((X + Z) * MU - 0.5 * MU**2) / (sig**2))   # (M, n_gh, N)
    w = w_unnorm / w_unnorm.sum(axis=2, keepdims=True)          # normalize over i

    # E[V | x_j, z_k] for each j,k
    Ev = np.sum(w * V[None, None, :], axis=2)                   # (M, n_gh)

    # Integrate over z with GH weights
    val = Ev @ p                                                # (M,)

    return val[0] if np.isscalar(x) else val

V_torch = torch.from_numpy(V).reshape(N, 1)

fig2, ax2 = plt.subplots(figsize=(5, 3), dpi=256)

# Static axis settings
ax2.set_ylim([-3, 1])
ax2.set_xlim([-4, 4])
ax2.set_ylabel(r"$\phi^{\alpha_k,\nu}(x) - C_{\alpha_k}(x)$")
ax2.set_xlabel(r"$x$")

# Create artists once with dummy initial data
mu0 = opt_mu[0]
mu0_torch = torch.from_numpy(mu0).reshape(N, 1)
phi0, _ = potentials(mu0_torch, V_torch)
phi0 = phi0[0].numpy().ravel()
P0 = P_vec(mu0, mu0)
J0 = 0.5 * mu0**2 - phi0 - mu0 * P0
xx0 = np.linspace(mu0.min(), mu0.max(), 256)
JJ0 = np.interp(xx0, mu0, J0)

scatter2 = ax2.scatter(mu0, J0)
line2, = ax2.plot(xx0, JJ0)

for k in range(n_iter):
    mu = opt_mu[k]
    mu_torch = torch.from_numpy(mu).reshape(N, 1)

    # Sinkhorn potentials
    phi, _ = potentials(mu_torch, V_torch)
    phi = phi[0].numpy().ravel()

    # Compute P(mu) with vectorized Gauss–Hermite
    P_mu = P_vec(mu, mu)

    # J(x) = 0.5 x^2 - φ(x) - x P(x)
    J = 0.5 * mu**2 - phi - mu * P_mu

    xx = np.linspace(mu.min(), mu.max(), 256)
    JJ = np.interp(xx, mu, J)

    # Update artists
    scatter2.set_offsets(np.column_stack((mu, J)))
    line2.set_data(xx, JJ)

    ax2.set_title(fr"$k = {k}$")

    fig2.savefig(
        f"potential-plot/fig{k}.jpeg",
        transparent=False,
        bbox_inches="tight",
    )