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
		pos = [0,0]
		Min1 = int(input('input the minimum1:'))
		Max1 = int(input('input the maximum1:'))
		width = int(input('width: '))
		indicator = 0
		for i in range(length):
			if int(xlist[i]) >= Min1 and indicator == 0:
				pos[0] = i
				indicator = 1
			elif int(xlist[i]) >= Max1 and indicator == 1:
				pos[1] = i
				break

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

		pos = [0,0]
		indicator = 0
		for i in range(np.size(xlist_template)):
			if int(xlist_template[i]) >= Min1 and indicator == 0:
				pos[0] = i
				indicator = 1
			elif int(xlist_template[i]) >= Max1 and indicator == 1:
				pos[1] = i
				break
		xlist_template_cut = xlist_template[pos[0]:(pos[1]+1)]
		ylist_template_cut = ylist_template[pos[0]:(pos[1]+1)]
		
		xlist_template_v = (xlist_template_cut - 5891)/5891*300000
		fit_xlist_v  = (fit_xlist - 5891)/5891*300000

		def log_likelihood(theta, xlist_template_v, ylist_template, fit_xlist_v, fit_ylist):
			v1, v2, w1, w2, r, A = theta
			p1 = (1-((xlist_template_v-v1)/w1)**2)>0
			P1 = (1-((xlist_template_v-v1)/w1)**2)*p1
			p2 = (1-((xlist_template_v-v2)/w2)**2)>0
			P2 = (1-((xlist_template_v-v2)/w2)**2)*p2
			kernel = P1 + r*P2
			conv_result = signal.convolve(ylist_template_cut,kernel,mode='same')*A
			tck = interpolate.splrep(xlist_template_v,conv_result,s=0)
			conv_result = interpolate.splev(fit_xlist_v,tck,der=0)
			return -0.5*np.sum((fit_ylist - conv_result)**2)

		'''
		vshift = -2996
		vsep = 6460
		v1 = vshift - 0.5*vsep
		v2 = vshift + 0.5*vsep
		v1 = 5890*(1+v1/300000)
		v2 = 5890*(1+v2/300000)
		w1 = 5890*1211/300000
		w2 = 5890*5116/300000
		r = 0.636
		A = 0.1
		v1 = 5800
		v2 = 6000
		guess = [v1,v2,w1,w2,r,A]
		bounds = [(5700,6100),(5700,6100),(0,400),(0,400),(0,np.inf),(0,np.inf)]
		nll = lambda *args: -log_likelihood(*args)
		''
		res = optimize.minimize(nll, guess, args=(xlist_template, ylist_template, fit_xlist, fit_ylist), bounds=bounds)

		theta = res.x
		v1, v2, w1, w2, r, A = theta
		print(theta)
		'''


		vshift = -2996
		vsep = 6460
		v1 = vshift - 0.5*vsep
		v2 = vshift + 0.5*vsep
		w1 = 1211
		w2 = 5116
		r = 0.636
		A = 0.027
		
		p1 = (1-((xlist_template_v-v1)/w1)**2)>0
		P1 = (1-((xlist_template_v-v1)/w1)**2)*p1
		p2 = (1-((xlist_template_v-v2)/w2)**2)>0
		P2 = (1-((xlist_template_v-v2)/w2)**2)*p2
		kernel = P1 + r*P2
		conv_result = signal.convolve(ylist_template_cut,kernel,mode='same')*A
		tck = interpolate.splrep(xlist_template_v,conv_result,s=0)
		conv_result1 = interpolate.splev(fit_xlist_v,tck,der=0)
		print(-0.5*np.sum((fit_ylist - conv_result1)**2))

		fig, axs = plt.subplots(3,2)
		axs[0,0].plot(xlist_template_cut,ylist_template_cut)
		axs[0,1].plot(xlist_template_cut,conv_result,c='b')
		axs[0,1].plot(fit_xlist,conv_result1,c='r')
		axs[1,0].plot(xlist_template_v,P1*A)
		axs[1,1].plot(xlist_template_v,r*P2*A)
		axs[2,0].plot(xlist_template_v,kernel*A)
		axs[2,1].plot(fit_xlist, fit_ylist, c = 'b', label = 'data')
		axs[2,1].plot(fit_xlist, conv_result1, c = 'r', label = 'conv_fit')
		axs[2,1].legend()
		plt.show()
		again = int(input('again?: '))
		if again != 1:
			break
#2003hv, figure in figure, edge