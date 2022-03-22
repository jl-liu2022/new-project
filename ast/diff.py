import numpy as np
import matplotlib.pyplot as plt

def ddG(x, A, x0, sigma):
	return A/sigma**2*np.exp(-(x-x0)**2/2/sigma**2)*(-1+(x-x0)**2/sigma**2)

def G(x, A, x0, sigma):
	return A*np.exp(-(x-x0)**2/2/sigma**2)

def fddG(x, shift_v, shift_w):
	return ddG(x, 1, 7155*shift_v, 7155*shift_w) + ddG(x, 0.24, 7172*shift_v, 7172*shift_w)

def fG(x, shift_v, shift_w):
	return G(x, 1, 7155*shift_v, 7155*shift_w) + G(x, 0.24, 7172*shift_v, 7172*shift_w)

shift_v = 1+2000/300000
shift_w = 7000/300000
xlist = np.linspace(6800,7800,1000)
fig, ax = plt.subplots(2,1,sharex=True)
ax[0].plot(xlist, fddG(xlist, shift_v, shift_w))
ax[1].plot(xlist, fG(xlist, shift_v, shift_w))
plt.show()