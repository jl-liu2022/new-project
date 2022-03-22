import matplotlib.pyplot as plt
import numpy as np
import emcee

def log_prob(x, mu, cov):
    diff = x - mu
    return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

ndim = 5
nwalkers = 32
np.random.seed(42)
means = np.random.rand(ndim)
cov = 0.5 - np.random.rand(ndim**2).reshape((ndim,ndim))
cov = np.triu(cov)
cov += cov.T - np.diag(cov.diagonal())
cov = np.dot(cov,cov)

p0 = np.random.rand(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[means, cov])

state = sampler.run_mcmc(p0,100)
sampler.reset()
sampler.run_mcmc(state,10000)
samples = sampler.get_chain(flat=True)
plt.hist(samples[:,0],100,color='k',histtype='step')
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$p(\theta_1)$")
plt.gca().set_yticks([])
plt.show()