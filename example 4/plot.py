import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
data = np.load("data.npz", allow_pickle=True)
mu = np.asarray(data["mu"])          # shape (n_iter, N_samp)
n_iter = int(data["n_iter"])
N = int(data["N_samp"])

# ---------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------
iters = np.arange(n_iter)
x = np.repeat(iters, N)              # (n_iter*N,)
y = mu.reshape(-1)                   # (n_iter*N,)

fig, ax = plt.subplots(figsize=(7, 5), dpi=256)
ax.scatter(x, y, color="C0", marker=".", s=6, label="empirical measure")

ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$\mu_k \sim \mathcal{U}\left(x^{(k)}_1,\dots,x^{(k)}_n\right)$")
ax.set_title(r'$J(\mu,x) = \int tan^{-1}(x-y) \mu(dy) + \frac{1}{2}(x-1)^2$')
ax.hlines(y=0, xmin=0, xmax=n_iter - 1, ls="--", color="C1", label="MFE")
ax.legend()

fig.savefig("ex4-3.7(i)-tangpi-touzi.jpeg", transparent=False, bbox_inches="tight")
plt.close(fig)