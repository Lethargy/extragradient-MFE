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
e = np.exp(1/sig**2)
pi = np.array([0.3, 0.7])

np.random.seed(0)

# initial V
V = np.sort(norm.rvs(size=2))
np.save('data/V.npy', V)

# Gauss–Hermite nodes / weights for N(0, σ²)
n_gh = 20
z, p = roots_hermite(n_gh)
z = np.sqrt(2.0) * sig * z        # nodes for N(0, σ²)
p = p / np.sqrt(np.pi)            # weights for N(0, σ²)

def H(y,M):
    e1 = pow(e, M[1] * y - 0.5 * M[1]**2)
    e0 = pow(e, M[0] * y - 0.5 * M[0]**2)
    return (pi[0] * V[0] * e0 + pi[1] * V[1] * e1) / (pi[0] * e0 + pi[1] * e1)

def dC(x,M):
    """Vectorized gradient ∇C_μ(x).

    x : scalar or 1D array of evaluation points
    """
    out = 0.0
    for i in range(n_gh):
        out += p[i] * (1 + x * z[i] / sig**2) * H(x + z[i], M)
    return out
    

# initial X
alpha = norm.rvs(size = 2); alpha.sort()
X = norm.rvs(size = 2); X.sort()
Y = norm.rvs(size = 2); Y.sort()

tol = 1e-15

for i in range(100):
    np.save(f"data/mu{i:02d}.npy", alpha)
    # first optimization
    T = interp1d(alpha, V, fill_value = 'extrapolate') # optimal map from mu to V
    dX = pi * (X - alpha - lam * (T(X) - [dC(x, alpha) for x in X]))

    while np.sqrt(dX @ dX) > tol:
        X = pav(X - r * dX)
        dX = pi * (X - alpha - lam * (T(X) - [dC(x, alpha) for x in X]))
    
    # second optimization
    beta = X.copy()
    T = interp1d(beta,V, fill_value = 'extrapolate') # optimal map from nu to V
    dY = pi * (Y - alpha - lam * (T(Y) - [dC(y,beta) for y in Y]))

    while np.sqrt(dY @ dY) > tol:
        Y = pav(Y - r * dY)
        dY = pi * (Y - alpha - lam * (T(Y) - [dC(y,beta) for y in Y]))
            
    alpha = Y.copy()
