import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import scipy.integrate as itg
import scipy.signal as signal
import matplotlib.collections as collections
from extinction import fitzpatrick99 
import scipy.signal as signal
from scipy import interpolate
import sys

spectrum_filename = input('spectrum_filename: ')
with open(spectrum_filename,'r') as f:
	xlist = []
	ylist = []
	start = int(input('start: '))
	for i in range(start):
		line = f.readline()
	while(line):
		a = line.split()
		xlist.append(float(a[0]))
		ylist.append(float(a[1]))
		line = f.readline()

xlist = np.array(xlist)
ylist = np.array(ylist)

redshift = float(sys.argv[1])
E_B_V = float(sys.argv[2])
R_V = float(sys.argv[3])

ylist = ylist*np.power(10,fitzpatrick99(xlist,R_V*E_B_V,R_V)/2.5)
ylist = ylist/np.max(ylist)
xlist = xlist / (1+redshift)

while(1):
	width = int(input('width: '))
	ylist_t = signal.savgol_filter(ylist,width,1)
	plt.plot(xlist, ylist, c = 'gray')
	plt.plot(xlist, ylist_t, c = 'black')
	plt.show()
	plt.savefig('tempt_figure.png')
	Min = float(input('Min: '))
	Max = float(input('Max: '))
	cut_xlist = xlist[np.argmax(xlist>Min):np.argmax(xlist>Max)]
	cut_ylist = ylist[np.argmax(xlist>Min):np.argmax(xlist>Max)]
	cut_ylist_t = ylist_t[np.argmax(xlist>Min):np.argmax(xlist>Max)]
	tck = interpolate.splrep(cut_xlist, cut_ylist_t)
	def curve(x):
		return interpolate.splev(x, tck)
	res = optimize.minimize(fun=curve, x0=np.array([6140]), bounds=optimize.Bounds(5950,6250))
	min_point = res.x
	print(min_point)
	v_Si = (min_point-6355)/6355*300000
	plt.plot(cut_xlist, cut_ylist, c = 'gray')
	plt.plot(cut_xlist, cut_ylist_t, c = 'black')
	plt.plot(cut_xlist, curve(cut_xlist), c='blue')
	plt.axvline(min_point)
	plt.show()
	plt.savefig('./tempt_figure.png')
	print('v_Si: ', v_Si)
	again = input('again?(y/n):')
	if again == 'n':
		break