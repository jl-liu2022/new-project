import matplotlib.pyplot as plt
import numpy as np

'''
xlist = []
ylist = []

filename = 'data'

with open(filename,'r') as f:
	start = 1
	for i in range(start):
		line = f.readline()
	while(line):
		a = line.split()
		xlist.append(float(a[0]))
		ylist.append(float(a[1]))
		line =f.readline()


xlist = np.array(xlist)
xlist = xlist * 0.1632 + 0.0152
ylist = np.array(ylist)
N = np.size(xlist)
x0 = xlist[0]
x1 = xlist[N-1]
xt = np.linspace(xlist[0], xlist[N-1], 100)
ipo3 = spi.splrep(xlist, ylist, k=3)
iy3 = spi.splev(xt, ipo3)

def fit(x):
	return spi.splev(x, ipo3)

guess = (0.322)
res = optimize.minimize(fit, guess)
xmin = res.x
print('B0: %f' %xmin)
ymin = fit(xmin)
ymax = ylist[0]
ytarget = (ymax+ymin)/2

def root(x):
	return fit(x) - ytarget

sol1 = optimize.root(root, 0.315)
sol2 = optimize.root(root, 0.325)
print('B1: %f' %sol1.x)
print('B2: %f' %sol2.x)
deltB = sol2.x - sol1.x
print('∆B: %f' %deltB)
'''
x1 = np.array([0.16,0.30,0.49,0.60,0.63,0.65,0.66,0.67,0.70,0.81,0.95,1.14])*0.1
Upp1 = np.array([18,24,62,136,162,178,180,178,144,64,30,18])
plt.title('reverse')
plt.xlabel('I [A]')
plt.ylabel('Upp [mV]')
plt.plot(x1, Upp1, c = 'b')
plt.scatter(x1, Upp1, c = 'r')
plt.show()

x1 = np.array([0.16,0.38,0.60,0.81,1.10])*0.1
Upp1 = np.array([14,14,14,14,14])
plt.title('syntropy')
plt.xlabel('I [A]')
plt.ylabel('Upp [mV]')
plt.plot(x1, Upp1, c = 'b')
plt.scatter(x1, Upp1, c = 'r')
plt.show()

x1 = np.array([0.18,0.36,0.66,0.89,1.18])*0.1
Upp1 = np.array([12,12,12,12,12])
plt.title('(a)')
plt.xlabel('I [A]')
plt.ylabel('Upp [mV]')
plt.plot(x1, Upp1, c = 'b')
plt.scatter(x1, Upp1, c = 'r')
plt.show()

x1 = np.array([0.18,0.53,0.55,0.58,0.60,0.62,0.66,0.71,0.76,0.82,0.86,0.92])*0.1
Upp1 = np.array([12,14,18,40,110,176,184,192,192,152,20,12])
plt.title('(b)')
plt.xlabel('I [A]')
plt.ylabel('Upp [mV]')
plt.plot(x1, Upp1, c = 'b')
plt.scatter(x1, Upp1, c = 'r')
plt.show()

x1 = np.array([0.18,0.20,0.29,0.35,0.36,0.37,0.38,0.39,0.40,0.41,0.42,0.43,0.46])*0.1
Upp1 = np.array([184,188,186,180,182,176,152,76,36,24,16,12,12])
plt.title('(c)')
plt.xlabel('I [A]')
plt.ylabel('Upp [mV]')
plt.plot(x1, Upp1, c = 'b')
plt.scatter(x1, Upp1, c = 'r')
plt.show()

