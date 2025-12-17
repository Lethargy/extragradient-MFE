import numpy as np
from scipy.special import roots_hermite
from scipy.stats import norm
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------
# One helper: PAV (isotonic) projection onto x1 <= ... <= xn, O(n)
# ---------------------------------------------------------------------
def pav(y: np.ndarray) -> np.ndarray:
    """L2 projection of y onto the monotone nondecreasing cone."""
    y = np.asarray(y, dtype=float)
    lvl = []
    wts = []
    for val in y:
        lvl.append(val)
        wts.append(1)
        while len(lvl) >= 2 and lvl[-2] > lvl[-1]:
            w = wts[-2] + wts[-1]
            avg = (wts[-2] * lvl[-2] + wts[-1] * lvl[-1]) / w
            lvl[-2] = avg
            wts[-2] = w
            lvl.pop(); wts.pop()
    out = np.empty_like(y)
    i = 0
    for avg, w in zip(lvl, wts):
        out[i:i+w] = avg
        i += w
    return out

# ---------------------------------------------------------------------
# Set parameters
# ---------------------------------------------------------------------
lam = 0.1
r = 0.75
sig = 0.5
pi = np.array([0.3, 0.7])          # component weights (same shape as X/alpha/Y)
tol = 1e-15

np.random.seed(0)

# Initial V (two-point value distribution here, but code supports general n)
V = np.sort(norm.rvs(size=2))
# np.save('data/V.npy', V)  # uncomment if you want this side effect

# Gauss–Hermite nodes / weights for N(0, σ²)
n_gh = 20
z, p = roots_hermite(n_gh)
z = np.sqrt(2.0) * sig * z
p = p / np.sqrt(np.pi)

inv_sig2 = 1.0 / (sig * sig)

def H(y: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Vectorized H(y, M). y may be scalar or array."""
    y = np.asarray(y, dtype=float)
    M = np.asarray(M, dtype=float)

    # pow(e, a) with e = exp(1/sig^2) so pow(e,a)=exp(a/sig^2)
    a1 = (M[1] * y - 0.5 * M[1] * M[1]) * inv_sig2
    a0 = (M[0] * y - 0.5 * M[0] * M[0]) * inv_sig2

    e1 = np.exp(a1)
    e0 = np.exp(a0)

    num = (pi[0] * V[0] * e0) + (pi[1] * V[1] * e1)
    den = (pi[0] * e0) + (pi[1] * e1)
    return num / den


def dC(x: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Vectorized gradient ∇C_μ(x, M) for x an array."""
    x = np.asarray(x, dtype=float)               # (m,)
    X = x[:, None]                               # (m,1)
    Z = z[None, :]                               # (1,n_gh)
    Y = X + Z                                    # (m,n_gh)

    # (1 + x z / σ²)
    factor = 1.0 + (X * Z) * inv_sig2            # (m,n_gh)
    Hy = H(Y, M)                                 # (m,n_gh)

    return (factor * Hy) @ p                     # (m,)


# initial variables (sorted)
alpha = np.sort(norm.rvs(size=2))
X     = np.sort(norm.rvs(size=2))
Y     = np.sort(norm.rvs(size=2))

tol2 = tol * tol
inner_max = 200_000  # safety cap
mu_hist = []

for it in range(100):
    mu_hist.append(alpha.copy())

    # -------------------------------------------------------------
    # First optimization: update X
    # -------------------------------------------------------------
    # Optimal map T from alpha to V (monotone 1D)
    T = interp1d(alpha, V, fill_value = 'extrapolate') # optimal map from alpha to L(V)
    dX = pi * (X - alpha - lam * (T(X) - dC(X, alpha)))

    k = 0
    while (dX @ dX > tol2) and (k < inner_max):
        X = pav(X - r * dX)
        dX = pi * (X - alpha - lam * (T(X) - dC(X, alpha)))
        k += 1

    # -------------------------------------------------------------
    # Second (extragradient) optimization: update Y using beta = X
    # -------------------------------------------------------------
    beta = X.copy()
    T = interp1d(beta,V, fill_value = 'extrapolate') # optimal map from beta to L(V)
    dY = pi * (Y - alpha - lam * (T(Y) - dC(Y, beta)))

    k = 0
    while (dY @ dY > tol2) and (k < inner_max):
        Y = pav(Y - r * dY)
        dY = pi * (Y - alpha - lam * (T(Y) - dC(Y, beta)))
        k += 1

    alpha = Y.copy()

mu_hist = np.stack(mu_hist)
np.savez(
    "data/run_001.npz",
    mu=mu_hist,
    V=V,
    # optional: store params too
    lam=lam,
    sig=sig,
    r=r,
    pi=pi
)