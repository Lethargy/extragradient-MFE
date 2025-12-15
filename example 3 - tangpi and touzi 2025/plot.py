import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
mu = [np.load('data/'+mu) for mu in sorted(os.listdir('data'))]
n_iter = len(mu)
N = len(mu[0])

# ---------------------------------------------------------------------
# Transport plot (X(v) vs v)
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize = (7,5), dpi = 256)

ax.scatter(np.repeat(0,N), mu[0], color = 'C0', marker = '.', label = 'empirical measure')

for i in range(1,n_iter):
    ax.scatter(np.repeat(i,N), mu[i], color = 'C0', marker = '.')

ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$\mu_k \sim \mathcal{U}(x_1,\dots,x_n)$')
ax.hlines(y = -0.5, xmin = 0, xmax = n_iter-1, ls = '--', color = 'C1', label = 'MFE')
ax.legend()

fig.savefig('plots/3.1i-tangpi-touzi.jpeg', transparent = False, bbox_inches = 'tight')