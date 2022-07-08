import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import scipy.integrate as itg
import scipy.signal as signal
from extinction import fitzpatrick99 

def Append(l1, l2):
	l3 = []
	for item in l1:
		l3.append(item)
	for item in l2:
		l3.append(item)
	return l3
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
		print(resolution)
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
	plt.title('%s at phase %s d' %(target_name, phase))
	plt.xlabel('Wavelength [Å]')
	plt.ylabel('Normalized flux')
	plt.yticks([])
	plt.plot(xlist,ylist)
	plt.show()

	Min1 = int(input('input the minimum1:'))
	Max1 = int(input('input the maximum1:'))
	Min2 = int(input('input the minimum2:'))
	Max2 = int(input('input the maximum2:'))

	'''optimize'''
	length = len(xlist)
	def fG(x, sigma, mu, A):
		return A*np.exp(-(x-mu)**2/2/sigma**2)

	width = int(input('width: '))
	xlist = xlist / (1+redshift)
	ylist = ylist*np.power(10,fitzpatrick99(xlist,R_V*E_B_V,R_V)/2.5)
	ylist = ylist * amp_y


	pos = [0,0,0,0]
	sig =  7000/2/300000*7155/(2*np.log(2))**(1/2)
	sub1 =  0
	sup1 =  float('inf')
	sub2 =  0
	sup2 =  upper_FWHM_Ni/2/300000*7378/(2*np.log(2))**(1/2)
	bound =  ([sub1,0.98,0.98,0,0,sub2],[sup1,1.02,1.02,float('inf'),float('inf'),sup2])
	guess = [sig,1,1,200,100,sig]
	internal = int(xlist[1]) - int(xlist[0])


	while(1):
		indicator = 0
		for i in range(length):
			if int(xlist[i]) >= Min1 and indicator == 0:
				pos[0] = i
				indicator = 1
			elif int(xlist[i]) >= Max1 and indicator == 1:
				pos[1] = i
				indicator = 2
			elif int(xlist[i]) >= Min2 and indicator == 2:
				pos[2] = i
				indicator = 3
			elif int(xlist[i]) >= Max2 and indicator == 3:
				pos[3] = i
				break

		print(xlist[pos[0]], xlist[pos[1]], xlist[pos[2]], xlist[pos[3]])
		cut_xlist = xlist[pos[0]:(pos[3]+1)]
		ylist_t = signal.savgol_filter(ylist,width,1)
		cut_ylist = ylist_t[pos[0]:(pos[3]+1)]
		print(len(cut_ylist))

		cut_ylist_f = []
		for i in range(len(cut_ylist)):
			cut_ylist_f.append(cut_ylist[i])
		cut_ylist_f = np.array(cut_ylist_f)

		x0 = cut_xlist[0]
		y0 = cut_ylist_f[0]
		x1 = cut_xlist[pos[3] - pos[0]]
		y1 = cut_ylist_f[pos[3] - pos[0]]
		

		fit_xlist = Append(cut_xlist[0:(pos[1] - pos[0]+1)],cut_xlist[(pos[2] - pos[0]):(pos[3] - pos[0]+1)])
		fit_ylist = Append(cut_ylist_f[0:(pos[1] - pos[0]+1)],cut_ylist_f[(pos[2] - pos[0]):(pos[3] - pos[0]+1)])
		y7 = (y0 - y1) / (x0-x1) * (fit_xlist - x0) + y0
		plt.scatter(fit_xlist, fit_ylist - y7, s = 5)
		plt.show()

		def fGs(x,s1,t1,t2,a,b,s2):
			return fG(x,s1,7155*t1,a)+fG(x,s1,7172*t1,0.24*a)+fG(x,s1,7388*t1,0.19*a)+fG(x,s1,7453*t1,0.31*a)+fG(x,s2,7378*t2,b)+fG(x,s2,7412*t2,0.31*b) + (y0 - y1) / (x0-x1) * (x - x0) + y0

		params,params_covariance=optimize.curve_fit(fGs,fit_xlist,fit_ylist,guess,maxfev=500000,bounds=bound)

		y7 = (y0 - y1) / (x0-x1) * (cut_xlist - x0) + y0
		ysimu = fGs(cut_xlist,params[0],params[1],params[2],params[3],params[4],params[5]) - y7
		y1 = fG(cut_xlist,params[0],7155*params[1],params[3])
		y2 = fG(cut_xlist,params[0],7172*params[1],0.24*params[3])
		y3 = fG(cut_xlist,params[0],7388*params[1],0.19*params[3])
		y4 = fG(cut_xlist,params[0],7453*params[1],0.31*params[3])
		y5 = fG(cut_xlist,params[5],7378*params[2],params[4])
		y6 = fG(cut_xlist,params[5],7412*params[2],0.31*params[4])


		plt.title('%s at phase %s d' %(target_name, phase))
		plt.xlabel('Rest Wavelength [Å]')
		plt.ylabel('Normalized flux')
		plt.yticks([])
		plt.plot(cut_xlist, cut_ylist_f - y7, color="blue", label="data")
		plt.plot(cut_xlist, ysimu, color="red",  label="best fit")
		plt.plot(cut_xlist, y1, color="purple", label="[Fe II]")
		plt.plot(cut_xlist, y2, color="purple")
		plt.plot(cut_xlist, y3, color="purple")
		plt.plot(cut_xlist, y4, color="purple")
		plt.plot(cut_xlist, y5, color="green", label="[Ni II]")
		plt.plot(cut_xlist, y6, color="green")
		plt.plot(cut_xlist[0:(pos[1] - pos[0]+1)], np.zeros(pos[1]-pos[0]+1)*min(cut_ylist_f - y7), c = 'y', label = 'fit region')
		plt.plot(cut_xlist[(pos[2] - pos[0]):(pos[3] - pos[0]+1)], np.zeros(pos[3]-pos[2]+1)*min(cut_ylist_f - y7), c = 'y')
		plt.plot(cut_xlist, y7, color="green",   label="continuum")
		plt.legend(loc='upper left')
		plt.show()

		def fGs_improve(x, s1, t1, a):
			return fG(x,s1,7155*t1,a)+fG(x,s1,7172*t1,0.24*a)+fG(x,s1,7388*t1,0.19*a)+fG(x,s1,7453*t1,0.31*a)

		ylist_tempt1 = cut_ylist_f - y7 - y5 - y6
		ylist_tempt2 = cut_ylist_f - y7 - y1 - y2 - y3 - y4
		'''
		plt.plot(cut_xlist, ylist_tempt1, label = 'data', c = 'b')
		plt.plot(cut_xlist, fGs_improve(cut_xlist, params[0], params[1], params[3]), label = 'fit', c = 'r')
		plt.legend(loc = 'upper left')
		plt.show()

		plt.plot(cut_xlist, ylist_tempt2, label = 'data', c = 'b')
		plt.plot(cut_xlist, y5+y6, label = 'fit', c = 'r')
		plt.legend(loc = 'upper left')
		plt.show()
		'''

		bound_i =  ([sub1,0.98,0],[sup1,1.02,float('inf')])
		guess_i = [sig,1,1]
		params_i, params_covariance_i = optimize.curve_fit(fGs_improve,cut_xlist,ylist_tempt1,guess_i,maxfev=500000,bounds=bound_i)

		'''
		plt.plot(cut_xlist, ylist_tempt1, label = 'data', c = 'b')
		plt.plot(cut_xlist, fGs_improve(cut_xlist, params_i[0], params_i[1], params_i[2]), label = 'fit', c = 'r')
		plt.legend(loc = 'upper left')
		plt.show()
		'''

		params[0] = params_i[0]
		params[1] = params_i[1]
		params[3] = params_i[2]

		'''
		plt.plot(cut_xlist, cut_ylist_f-y7, label = 'data', c = 'b')
		plt.plot(cut_xlist, fGs_improve(cut_xlist, params[0], params[1], params[3])+y5+y6, label = 'fit_f', c = 'black')
		plt.plot(cut_xlist, ysimu, label = 'fit_i0', c = 'yellow')
		plt.plot(cut_xlist, fGs(cut_xlist,params[0],params[1],params[2],params[3],params[4],params[5]) - y7, label = 'fit_i', c = 'purple')
		plt.legend(loc = 'upper left')
		plt.show()
		'''

		FWHM_Fe = params[0] * 2 * 300000 / 7155 * np.sqrt(2*np.log(2))
		FWHM_Ni = params[5] * 2 * 300000 / 7378 * np.sqrt(2*np.log(2))
		vshift_Fe = (params[1] - 1) * 300000 
		vshift_Ni = (params[2] - 1) * 300000
		flux_ratio = params[4] * params[5] / params[3] / params[0]
		high_ratio = params[4] / params[3]
		print('FWHM_Fe: %f, FWHM_Ni: %f, vshift_Fe: %f, vshift_Ni: %f, flux_ratio Ni/Fe: %f, high_ratio Ni/Fe: %s' % (FWHM_Fe,FWHM_Ni, vshift_Fe, vshift_Ni, flux_ratio, high_ratio))


		def f1(x):
			return params[4] * params[5] / params[3] / params[0] / 4.9 /np.exp(0.28*1.60217662/1.380694/10**(-4)/x)

		v, err = itg.quad(f1, 3000, 8000)
		ratio_ave_bef = v / 5000

		print('n_Ni/n_Fe = %f' % (ratio_ave_bef * 1.8))
		
		y1 = fG(cut_xlist,params[0],7155*params[1],params[3])
		y2 = fG(cut_xlist,params[0],7172*params[1],0.24*params[3])
		y3 = fG(cut_xlist,params[0],7388*params[1],0.19*params[3])
		y4 = fG(cut_xlist,params[0],7453*params[1],0.31*params[3])
		y5 = fG(cut_xlist,params[5],7378*params[2],params[4])
		y6 = fG(cut_xlist,params[5],7412*params[2],0.31*params[4])
		ysimu = y1 + y2 + y3 + y4 + y5 + y6

		plt.title('%s  +%sd' %(target_name, phase))
		plt.xlabel('Rest Wavelength [Å]')
		plt.ylabel('Normalized flux')
		plt.yticks([])
		plt.plot(xlist[(pos[0]-100):(pos[3]+100+1)], ylist_t[(pos[0]-100):(pos[3]+100+1)], color="blue", label="data")
		plt.plot(cut_xlist, ysimu+y7, color="red",  label="best fit")
		plt.plot(cut_xlist, y1+y7, color="purple", label="[Fe II]")
		plt.plot(cut_xlist, y2+y7, color="purple")
		plt.plot(cut_xlist, y3+y7, color="purple")
		plt.plot(cut_xlist, y4+y7, color="purple")
		plt.plot(cut_xlist, y5+y7, color="green", label="[Ni II]")
		plt.plot(cut_xlist, y6+y7, color="green")
		plt.plot(cut_xlist[0:(pos[1] - pos[0]+1)], np.zeros(pos[1]-pos[0]+1)*min(cut_ylist_f), c = 'y', label = 'fit region')
		plt.plot(cut_xlist[(pos[2] - pos[0]):(pos[3] - pos[0]+1)], np.zeros(pos[3]-pos[2]+1)*min(cut_ylist_f), c = 'y')
		plt.plot(cut_xlist, y7, color="black",   label="continuum")
		plt.legend(loc='upper left')
		plt.show()

		plt.title('%s  +%sd' %(target_name, phase))
		plt.xlabel('Rest Wavelength [Å]')
		plt.ylabel('Normalized flux')
		plt.yticks([])
		plt.plot(xlist[pos[0]:(pos[3]+1)], ylist[pos[0]:(pos[3]+1)] - y7, color="blue", label="data")
		plt.plot(cut_xlist, ysimu, color="red",  label="best fit")
		plt.plot(cut_xlist, y1, color="purple", label="[Fe II]")
		plt.plot(cut_xlist, y2, color="purple")
		plt.plot(cut_xlist, y3, color="purple")
		plt.plot(cut_xlist, y4, color="purple")
		plt.plot(cut_xlist, y5, color="green", label="[Ni II]")
		plt.plot(cut_xlist, y6, color="green")
		plt.plot(cut_xlist[0:(pos[1] - pos[0]+1)], np.zeros(pos[1]-pos[0]+1), c = 'y', label = 'fit region')
		plt.plot(cut_xlist[(pos[2] - pos[0]):(pos[3] - pos[0]+1)], np.zeros(pos[3]-pos[2]+1), c = 'y')

		plt.legend(loc='upper left')
		plt.show()
		'''
		plt.plot(cut_xlist, cut_ylist, color="blue", label="obser")
		plt.plot(cut_xlist, ysimu + y7, color="red",  label="simu")
		plt.plot(cut_xlist, y1, color="yellow",  label="7155")
		plt.plot(cut_xlist, y2, color="cyan",  label="7172")
		plt.plot(cut_xlist, y3, color="magenta",  label="7388")
		plt.plot(cut_xlist, y4, color="black",  label="7453")
		plt.plot(cut_xlist, y5, color="green",   label="7378")
		plt.plot(cut_xlist, y6, color="green",   label="7412")

		plt.plot(cut_xlist, y7, color="green",   label="continuum")
		plt.legend(loc='upper left')
		plt.show()
		'''
		q = int(input('input 1 to continue, 2 to save and quit, 3 to quit:'))
		if q == 2:
			if (Min2 - Max1) >= 10:
				save_as = '_IMG.dat'
			else:
				save_as = '_MG.dat'
			with open('/Users/pro/python/ast/result_data0'+save_as,'a') as f:
				f.writelines('%s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' %(target_name, phase, delta15, U_delta15, redshift, E_B_V, vSi, U_vSi, ratio_ave_bef * 1.8, vshift_Fe, vshift_Ni, FWHM_Fe, FWHM_Ni, flux_ratio, high_ratio))
				g = 1
				break
		elif q == 3:
			g = 0
			break
		change_Min1 = int(input('input change_Min1:'))
		change_Max1 = int(input('input change_Max1:'))
		change_Min2 = int(input('input change_Min2:'))
		change_Max2 = int(input('input change_Max2:'))
		upper_FWHM_Ni = float(input('the upper limit of FWHM_Ni'))
		sup2 =  upper_FWHM_Ni/2/300000*7378/(2*np.log(2))**(1/2)
		bound =  ([sub1,0.98,0.98,0,0,sub2],[sup1,1.02,1.02,float('inf'),float('inf'),sup2])
		width = int(input('width: '))
		if change_Min1 == 0 and change_Max1 == 0 and change_Min2 == 0 and change_Max2 and upper_FWHM_Ni == 0:
			break
		Min1 += change_Min1
		Max1 += change_Max1
		Min2 += change_Min2
		Max2 += change_Max2

	if g == 1:
		#Possion
		cut_xlist = xlist[pos[0]:(pos[3]+1)]
		cut_ylist = ylist[pos[0]:(pos[3]+1)]

		fit_xlist = Append(cut_xlist[0:(pos[1] - pos[0]+1)],cut_xlist[(pos[2] - pos[0]):(pos[3] - pos[0]+1)])
		fit_ylist = Append(cut_ylist[0:(pos[1] - pos[0]+1)],cut_ylist[(pos[2] - pos[0]):(pos[3] - pos[0]+1)])

		x0 = cut_xlist[0]
		y0 = cut_ylist[0]
		x1 = cut_xlist[pos[3] - pos[0]]
		y1 = cut_ylist[pos[3] - pos[0]]	

		params,params_covariance=optimize.curve_fit(fGs,fit_xlist,fit_ylist,guess,maxfev=500000,bounds=bound)

		y7 = (y0 - y1) / (x0-x1) * (cut_xlist - x0) + y0
		y5 = fG(cut_xlist,params[5],7378*params[2],params[4])
		y6 = fG(cut_xlist,params[5],7412*params[2],0.31*params[4])

		ylist_tempt = cut_ylist - y7 - y5 - y6

		bound_i =  ([sub1,0.98,0],[sup1,1.02,float('inf')])
		guess_i = [sig,1,1]
		params_i, params_covariance_i = optimize.curve_fit(fGs_improve,cut_xlist,ylist_tempt,guess_i,maxfev=500000,bounds=bound_i)

		params[0] = params_i[0]
		params[1] = params_i[1]
		params[3] = params_i[2]
		
		f_ratio = params[4] * params[5] / params[3] / params[0]
		h_ratio = params[4] / params[3]
		'''
		if np.abs(f_ratio/flux_ratio[0]) < 2 and np.abs(f_ratio/flux_ratio[0]) > 0.1 and np.abs(h_ratio/high_ratio[0]) < 3 and np.abs(h_ratio/high_ratio[0]) > 0.1:
		'''	

		Up_FWHM_Fe = FWHM_Fe - params[0] * 2 * 300000 / 7155 * np.sqrt(2*np.log(2))
		Up_vshift_Fe = vshift_Fe - (params[1] - 1) * 300000
		Up_FWHM_Ni = FWHM_Ni - params[5] * 2 * 300000 / 7378 * np.sqrt(2*np.log(2))
		Up_vshift_Ni = vshift_Ni - (params[2] - 1) * 300000
		Up_flux_ratio = flux_ratio - params[4] * params[5] / params[3] / params[0]
		Up_high_ratio = high_ratio - params[4]/params[3]
		#environment
		vshift_Fe = [vshift_Fe]
		vshift_Ni = [vshift_Ni]
		FWHM_Fe = [FWHM_Fe]
		FWHM_Ni = [FWHM_Ni]
		flux_ratio = [flux_ratio]
		high_ratio = [high_ratio]
		k = 0
		while(k < 1000):
			print(k)
			bluet1 = np.random.rand()*20 - 10 + Min1
			if (Min2 - Max1) >= 10:
				redt1 = np.random.rand()*20 - 10 + Max1
				bluet2 = np.random.rand()*20 - 10 + Min2
			else:
				redt1 = Max1
				bluet2 = Min2
			redt2 = np.random.rand()*20 - 10 + Max2
			n_min = 1
			indicator = 0
			for j in range(length):
				if int(xlist[j]) >= bluet1 and indicator == 0:
					pos[0] = j
					indicator = 1
				elif int(xlist[j]) >= redt1 and indicator == 1:
					pos[1] = j
					indicator = 2
				elif int(xlist[j]) >= bluet2 and indicator == 2:
					pos[2] = j
					indicator = 3
				elif int(xlist[j]) >= redt2 and indicator == 3:
					pos[3] = j
					break

			cut_xlist = xlist[pos[0]:(pos[3]+1)]
			cut_ylist = ylist_t[pos[0]:(pos[3]+1)]

			fit_xlist = Append(cut_xlist[0:(pos[1] - pos[0]+1)],cut_xlist[(pos[2] - pos[0]):(pos[3] - pos[0]+1)])
			fit_ylist = Append(cut_ylist[0:(pos[1] - pos[0]+1)],cut_ylist[(pos[2] - pos[0]):(pos[3] - pos[0]+1)])

			x0 = cut_xlist[0]
			y0 = cut_ylist[0]
			x1 = cut_xlist[pos[3] - pos[0]]
			y1 = cut_ylist[pos[3] - pos[0]]	

			params,params_covariance=optimize.curve_fit(fGs,fit_xlist,fit_ylist,guess,maxfev=500000,bounds=bound)

			y7 = (y0 - y1) / (x0-x1) * (cut_xlist - x0) + y0
			y5 = fG(cut_xlist,params[5],7378*params[2],params[4])
			y6 = fG(cut_xlist,params[5],7412*params[2],0.31*params[4])

			ylist_tempt = cut_ylist - y7 - y5 - y6

			bound_i =  ([sub1,0.98,0],[sup1,1.02,float('inf')])
			guess_i = [sig,1,1]
			params_i, params_covariance_i = optimize.curve_fit(fGs_improve,cut_xlist,ylist_tempt,guess_i,maxfev=500000,bounds=bound_i)

			params[0] = params_i[0]
			params[1] = params_i[1]
			params[3] = params_i[2]
			
			f_ratio = params[4] * params[5] / params[3] / params[0]
			h_ratio = params[4] / params[3]
			'''
			if np.abs(f_ratio/flux_ratio[0]) < 2 and np.abs(f_ratio/flux_ratio[0]) > 0.1 and np.abs(h_ratio/high_ratio[0]) < 3 and np.abs(h_ratio/high_ratio[0]) > 0.1:
			'''	
			FWHM_Fe.append(params[0] * 2 * 300000 / 7155 * np.sqrt(2*np.log(2)))
			FWHM_Ni.append(params[5] * 2 * 300000 / 7378 * np.sqrt(2*np.log(2)))
			vshift_Fe.append((params[1] - 1) * 300000) 
			vshift_Ni.append((params[2] - 1) * 300000)
			flux_ratio.append(params[4] * params[5] / params[3] / params[0])
			high_ratio.append(params[4]/params[3])
			k += 1
		Ue_FWHM_Fe = np.std(FWHM_Fe, ddof = 1)
		Ue_vshift_Fe = np.std(vshift_Fe, ddof = 1)
		Ue_FWHM_Ni = np.std(FWHM_Ni, ddof = 1)
		Ue_vshift_Ni = np.std(vshift_Ni, ddof = 1)
		Ue_flux_ratio = np.std(flux_ratio, ddof = 1)
		Ue_high_ratio = np.std(high_ratio, ddof = 1)

		U_FWHM_Fe = np.sqrt(Up_FWHM_Fe**2 + Ue_FWHM_Fe**2)
		U_vshift_Fe = np.sqrt(Up_vshift_Fe**2 + Ue_vshift_Fe**2)
		U_FWHM_Ni = np.sqrt(Up_FWHM_Ni**2 + Ue_FWHM_Ni**2)
		U_vshift_Ni = np.sqrt(Up_vshift_Ni**2 + Ue_vshift_Ni**2)
		U_flux_ratio = np.sqrt(Up_flux_ratio**2 + Ue_flux_ratio**2)
		U_high_ratio = np.sqrt(Up_high_ratio**2 + Ue_high_ratio**2)
		print('U_FWHM_Fe: %s, U_vshift_Fe: %s, U_FWHM_Ni: %s, U_vshift_Ni: %s, U_flux_ratio: %s, U_high_ratio: %s' %(U_FWHM_Fe, U_vshift_Fe, U_FWHM_Ni, U_vshift_Ni, U_flux_ratio, U_high_ratio))
		with open('/Users/pro/python/ast/Uncertenty'+save_as,'a') as f:
				#target phase delta15 n_Ni/n_Fe v_Fe v_Ni FWHM_Fe FWHM_Ni
				f.writelines('%s %s %s %s %s %s %s %s\n' %(target_name, phase, U_FWHM_Fe, U_vshift_Fe, U_FWHM_Ni, U_vshift_Ni, U_flux_ratio, U_high_ratio))
		g = 0
