import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import scipy.integrate as itg
import scipy.signal as signal
from scipy import interpolate
from extinction import fitzpatrick99 

def Append(l1, l2):
	l3 = []
	for item in l1:
		l3.append(item)
	for item in l2:
		l3.append(item)
	return np.array(l3)

def Cut(xlist, min, max):
	pos = [0,0]
	index = 0
	for i in range(len(xlist)):
		if xlist[i] >= min and index == 0:
			pos[0] = i
			index = 1
		if xlist[i] >= max and index ==1:
			pos[1] = i
			break
	return pos

def w_to_v_rela(wavelength, center):
	velocity = (wavelength**2 - center**2)/(wavelength**2 + center**2)*300000
	return velocity

def w_to_v_norm(wavelength, center):
	velocity = (wavelength - center)/center*300000
	return velocity

#template
with open('/Users/pro/python/spectra_data/paper/SN1999by/SN1999by.dat','r') as f:
	line = f.readline()
	a = line.split()
	redshift = float(a[1])
	line = f.readline()
	a = line.split()
	E_B_V = float(a[1])
	line = f.readline()
	a = line.split()
	R_V = float(a[1])
	xlist_template = []
	ylist_template = []
	line = f.readline()
	while line:
		a = line.split()
		xlist_template.append(float(a[0]))
		ylist_template.append(float(a[1]))
		line = f.readline()
print(redshift)
xlist_template = np.array(xlist_template)/(1+redshift)
ylist_template = np.array(ylist_template)*np.power(10,fitzpatrick99(xlist_template,R_V*E_B_V,R_V)/2.5)

plt.plot(xlist_template,ylist_template)
plt.show()

pos_temp1 = Cut(xlist_template, 5700, 6100)
pos_temp2 = Cut(xlist_template, 6065, 6335)


xlist_template1 = xlist_template[pos_temp1[0]: pos_temp1[1]]
ylist_template1 = ylist_template[pos_temp1[0]: pos_temp1[1]]
xlist_template2 = xlist_template[pos_temp2[0]: pos_temp2[1]]
ylist_template2 = ylist_template[pos_temp2[0]: pos_temp2[1]]

'''
open file
'''
while(1):	
	filename = input('input the name of the data file: ')
	for i in range(len(filename)):
		if filename[i] == '/':
			target_name = filename[:i]
	with open('/Users/pro/python/spectra_data/paper/' + filename,'r') as f:
		line = f.readline()
		a = line.split()
		phase = int(a[1])
		line = f.readline()
		a = line.split()
		delta15 = float(a[1])
		U_delta15 = float(a[2])
		line = f.readline()
		a = line.split()
		E_B_V = float(a[1])
		line = f.readline()
		a = line.split()
		R_V = float(a[1])
		line = f.readline()
		a = line.split()
		redshift = float(a[1])
		line = f.readline()
		a = line.split()
		amp_y = float(a[1])
		line = f.readline()
		a = line.split()
		resolution = float(a[1])
		line = f.readline()
		a = line.split()
		vSi = float(a[1])
		U_vSi = float(a[2])
		upper_FWHM_Ni = 13000
		print(resolution, redshift, E_B_V, R_V)
		xlist = []
		ylist = []
		line = f.readline()
		while line:
			a = line.split()
			xlist.append(float(a[0]))
			ylist.append(float(a[1]))
			line = f.readline()
	xlist = np.array(xlist)
	ylist = np.array(ylist)

	'''
	cut
	'''
	'''
	min = 6000
	max = 9000
	i=0
	pos = [0,0]
	length = len(xlist)
	for i in range(length):
		if int(xlist[i]) == min:
			pos[0] = i
		if int(xlist[i]) == max:
			pos[1] = i
			break
	'''
	

	'''optimize'''
	length = len(xlist)

	fig, axs = plt.subplots(2,1,sharex=True)
	axs[0].set_title('template')
	axs[0].plot(xlist_template,ylist_template)
	axs[1].set_title('target')
	axs[1].plot(xlist,ylist)
	plt.show()

	

	xlist = xlist / (1+redshift)
	ylist = ylist*np.power(10,fitzpatrick99(xlist,R_V*E_B_V,R_V)/2.5)
	ylist = ylist / np.max(ylist)

	while(1):
		Min1 = int(input('input the minimum1:'))
		Max1 = int(input('input the maximum1:'))
		width = int(input('width: '))
		pos = Cut(xlist, Min1, Max1)

		print(xlist[pos[0]], xlist[pos[1]])
		cut_xlist = xlist[pos[0]:(pos[1]+1)]
		if width != 1:
			ylist_t = signal.savgol_filter(ylist,width,1)
		if width == 1:
			ylist_t = np.array(ylist)
		cut_ylist = ylist_t[pos[0]:(pos[1]+1)]
		print(len(cut_ylist))
		
		fit_xlist = np.array(cut_xlist)
		fit_ylist = np.array(cut_ylist)
		
		def log_likelihood(theta, xlist_template1, ylist_template1, fit_xlist, fit_ylist):
			v1, v2, w1, w2, r, A = theta
			xlist_template1_v = w_to_v_norm(xlist_template1, 5900)
			p1 = (1-((xlist_template1_v-v1)/w1)**2)>0
			P1 = (1-((xlist_template1_v-v1)/w1)**2)*p1
			p2 = (1-((xlist_template1_v-v2)/w2)**2)>0
			P2 = (1-((xlist_template1_v-v2)/w2)**2)*p2
			kernel = P1 + r*P2

			conv_result1 = signal.convolve(ylist_template1,kernel,mode='same')*A
			tck = interpolate.splrep(xlist_template1, conv_result1, s=0)
			conv_result = interpolate.splev(fit_xlist,tck,der=0)
			return np.sum((fit_ylist - conv_result)**2)

		


		#vshift = -2996
		#vsep = 6460
		#w1 = 1211
		#w2 = 5116
		vshift = -3500
		vsep = 6000
		v1 = vshift - 0.5*vsep
		v2 = vshift + 0.5*vsep
		w1 = 2000
		w2 = 4000
		r = 0.636
		A = 0.027
		
		guess = [v1,v2,w1,w2,r,A]
		bounds = [(-10000,10000),(-10000,10000),(0,20000),(0,20000),(0,np.inf),(0,np.inf)]
		#nll = lambda *args: -log_likelihood(*args)

		res = optimize.minimize(log_likelihood, guess, args=(xlist_template1, ylist_template1, fit_xlist, fit_ylist), bounds=bounds)

		theta = res.x
		v1, v2, w1, w2, r, A = theta
		print(theta)
		 

		xlist_template1_v = w_to_v_norm(xlist_template1, 5900)
		p1 = (1-((xlist_template1_v-v1)/w1)**2)>0
		P1 = (1-((xlist_template1_v-v1)/w1)**2)*p1
		p2 = (1-((xlist_template1_v-v2)/w2)**2)>0
		P2 = (1-((xlist_template1_v-v2)/w2)**2)*p2
		kernel = P1 + r*P2

		conv_result1 = signal.convolve(ylist_template1,kernel,mode='same')*A
		tck = interpolate.splrep(xlist_template1, conv_result1, s=0)
		conv_result = interpolate.splev(fit_xlist,tck,der=0)

		pos_data = Cut(xlist, 5700, 6100)
		xlist_data = xlist[pos_data[0]:pos_data[1]]
		ylist_data = ylist[pos_data[0]:pos_data[1]]

		fig, axs = plt.subplots(2,1)
		axs[0].set_xlabel('Wavelength [$\\rm \\AA$]')
		axs[0].set_ylabel('Scaled Flux')
		axs[0].plot(xlist_template1,ylist_template1)
		axs[1].set_xlabel('wavelength [$\\rm \\AA$]')
		axs[1].set_ylabel('Scaled Flux')
		axs[1].plot(xlist_data, ylist_data, c = 'gray', label = 'data')
		axs[1].plot(xlist_template1,conv_result1, c = 'b', label = 'conv_result')
		axs[1].plot(fit_xlist, conv_result, c = 'r', label = 'fit region')
		axs[1].legend(loc = 'upper left')
		axins = axs[1].inset_axes([0.75,0.75,0.24,0.24])
		axins.set_yticklabels([])
		axins.set_xlabel('Velocity [km s$^{-1}$]')
		axins.plot(xlist_template1_v,kernel*A)

		plt.show()
		again = int(input('again?: '))
		if again != 1:
			break
#2003hv, figure in figure, edge