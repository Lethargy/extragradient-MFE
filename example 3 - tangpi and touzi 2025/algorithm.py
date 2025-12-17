import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------
# Helper: PAV (isotonic) projection onto x1 <= ... <= xn, O(n)
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

N_samp = 100
lam = 0.1
r = 0.5
n_iter = 40

np.random.seed(0)

X = np.sort(norm.rvs(size=N_samp))
Y = np.sort(norm.rvs(size=N_samp))
alpha = np.sort(norm.rvs(size=N_samp))

tol = 1e-6      # stopping tolerance
inner_max = 200_000  # safety cap (avoid infinite loops)

# small scratch buffers to avoid repeated allocations
cos_tmp = np.empty(N_samp)
sin_tmp = np.empty(N_samp)

# history of mu
mu_hist = []

for i in range(n_iter):
    mu_hist.append(alpha.copy())

    # -----------------------------------------------------------------
    # first optimization: update X
    # -----------------------------------------------------------------
    k = 0
    while k < inner_max:
        cX = np.cos(X).mean()
        sX = np.sin(X).mean()

        np.cos(alpha, out=cos_tmp)
        np.sin(alpha, out=sin_tmp)
        interaction = cX * cos_tmp + sX * sin_tmp

        dX = (X - alpha) + lam * (interaction + 2.0 * X)
        if dX @ dX <= tol:
            break

        X = pav(X - r * dX)
        k += 1

    # -----------------------------------------------------------------
    # second (extragradient) optimization: update Y with beta = X fixed
    # -----------------------------------------------------------------
    beta = X.copy()

    # precompute trig(beta) once (beta is constant in this inner loop)
    np.cos(beta, out=cos_tmp)
    cos_beta = cos_tmp.copy()
    np.sin(beta, out=sin_tmp)
    sin_beta = sin_tmp.copy()

    k = 0
    while k < inner_max:
        cY = np.cos(Y).mean()
        sY = np.sin(Y).mean()

        interaction = cY * cos_beta + sY * sin_beta

        dY = (Y - alpha) + lam * (interaction + 2.0 * Y)
        if dY @ dY <= tol:
            break

        Y = pav(Y - r * dY)
        k += 1

    alpha = Y.copy()
    
np.savez(
    "data/data.npz",
    mu=mu_hist,
    lam=lam,
    N_samp = N_samp,
    n_iter = n_iter,
    r=r
)
