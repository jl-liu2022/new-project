import numpy as np
from scipy import stats

def kendalltau_err(x, y, xerr, yerr):
    N = 11
    tau = []
    p = []
    for i in range(N):
        x_t = x + np.random.normal(scale=xerr)
        y_t = y + np.random.normal(scale=yerr)
        tau_t, p_t = stats.kendalltau(x_t, y_t)
        tau.append(tau_t)
        p.append(p_t)
    return tau, p

x = np.linspace(0,10,11)
y = x+1
xerr = 2*np.ones(11)
yerr = 2*np.ones(11)

tau, p = kendalltau_err(x, y, xerr, yerr)
tau_sort = np.percentile(tau, [16, 50, 84])
p_sort = np.percentile(p, [16, 50, 84])

print(tau)
print(tau_sort)
print(p)
print(p_sort)
