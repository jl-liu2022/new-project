import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import scipy.integrate as itg
import scipy.signal as signal
import matplotlib.collections as collections
from extinction import fitzpatrick99 
from tqdm import tqdm

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

def read_list(filename, start):
	with open(filename,'r') as f:
		for i in range (start):
			line = f.readline()
		NameList = []
		PhaseList = []
		Min1List = []
		Max1List = []
		Min2List = []
		Max2List = []
		Min3List = []
		Max3List = []
		Min4List = []
		Max4List = []
		WidthList = []
		ESList = []
		while line:
			a = line.split()
			NameList.append(a[0])
			PhaseList.append(a[1])
			Min1List.append(a[7])
			Max1List.append(a[8])
			Min2List.append(a[9])
			Max2List.append(a[10])
			Min3List.append(a[11])
			Max3List.append(a[12])
			Min4List.append(a[13])
			Max4List.append(a[14])
			WidthList.append(a[15])
			ESList.append(a[16])
			line = f.readline()
	return NameList, PhaseList, Min1List, Max1List, Min2List, Max2List,Min3List, Max3List, Min4List, Max4List, WidthList, ESList

list_name = 'Uncertenty_IMG.dat'
NameList, PhaseList, Min1List, Max1List, Min2List, Max2List,Min3List, Max3List, Min4List, Max4List, WidthList, ESList = read_list(list_name, 1)
list_size = np.size(NameList)

todo_name = 'Nothing'
initialize = False
for n in range(list_size):
	if initialize:
		phase = input('phase: ')
		filename = todo_name + '/' + todo_name + '_' + phase +'.dat'
		k = 0
		while(1):
			if todo_name[k] == 'N':
				break
			k += 1
		target_name = todo_name[0:(k+1)] + ' ' + todo_name[(k+1):]
	else:
		filename = NameList[n] + '/' + NameList[n] + '_' + PhaseList[n] +'.dat'
		k = 0
		while(1):
			if NameList[n][k] == 'N':
				break
			k += 1
		target_name = NameList[n][0:(k+1)] + ' ' + NameList[n][(k+1):]

	with open('paper/' + filename,'r') as f:
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
	

	

	'''optimize'''
	length = len(xlist)
	def fG(x, sigma, mu, A):
		return A*np.exp(-(x-mu)**2/2/sigma**2)
	
	'''
	fig, ax = plt.subplots()
	ax.set_title('%s + %sd' %(target_name, phase))
	ax.set_xlabel('Wavelength [Å]')
	ax.set_ylabel('Scaled Flux')
	ax.set_yticks([])
	ax.plot(xlist,ylist,color = 'b')
	

	collection = collections.BrokenBarHCollection.span_where(np.linspace(6800,7800,100),ymin=0,ymax=0.33,where=np.ones(100)>0,facecolor='gray',alpha=0.5)
	ax.add_collection(collection)

	collection = collections.BrokenBarHCollection.span_where(np.linspace(5700,6300,100),ymin=0,ymax=0.11,where=np.ones(100)>0,facecolor='gray',alpha=0.5)
	ax.add_collection(collection)

	collection = collections.BrokenBarHCollection.span_where(np.linspace(4500,4860,100),ymin=0,ymax=1.22,where=np.ones(100)>0,facecolor='gray',alpha=0.5)
	ax.add_collection(collection)

	ax.text(6600,0.34,'[Fe II] + [Ni II]')
	ax.text(5700,0.12,'[Co III]')
	ax.text(4300,1.23,'[Fe III]')

	plt.show()
	'''	
	if initialize:
		Min1 = 6800
		Max1 = 7000
		Min2 = 7000
		Max2 = 7300
		Min3 = 7300
		Max3 = 7500
		Min4 = 7500
		Max4 = 7800
		width = 1
	else:
		Min1 = int(Min1List[n])
		Max1 = int(Max1List[n])
		Min2 = int(Min2List[n])
		Max2 = int(Max2List[n])
		Min3 = int(Min3List[n])
		Max3 = int(Max3List[n])
		Min4 = int(Min4List[n])
		Max4 = int(Max4List[n])
		width = int(WidthList[n])

	ylist = ylist*np.power(10,fitzpatrick99(xlist,R_V*E_B_V,R_V)/2.5)
	xlist = xlist / (1+redshift)
	

	

	pos = [0,0,0,0,0,0,0,0]
	sig =  7000/2/300000/(2*np.log(2))**(0.5)      
	sub1 =  0
	sup1 =  float('inf')
	sub2 =  0
	sup2 =  upper_FWHM_Ni/2/300000/(2*np.log(2))**(1/2)
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
				indicator = 4
			elif int(xlist[i]) >= Min3 and indicator == 4:
				pos[4] = i
				indicator = 5
			elif int(xlist[i]) >= Max3 and indicator == 5:
				pos[5] = i
				indicator = 6
			elif int(xlist[i]) >= Min4 and indicator == 6:
				pos[6] = i
				indicator = 7
			elif int(xlist[i]) >= Max4 and indicator == 7:
				pos[7] = i
				break
		print(np.size(ylist))
		print(pos)
		ylist_norm = ylist / np.max(ylist[pos[0]:(pos[7]+1)])
		cut_xlist = xlist[pos[0]:(pos[7]+1)]
		if width != 1:
			ylist_t = signal.savgol_filter(ylist_norm,width,1)
		if width == 1:
			ylist_t = np.array(ylist_norm)
		cut_ylist = ylist_t[pos[0]:(pos[7]+1)]
		print(len(cut_ylist))

		cut_ylist_f = []
		for i in range(len(cut_ylist)):
			cut_ylist_f.append(cut_ylist[i])
		cut_ylist_f = np.array(cut_ylist_f)

		x0 = cut_xlist[0]
		y0 = cut_ylist_f[0]
		x1 = cut_xlist[pos[7] - pos[0]]
		y1 = cut_ylist_f[pos[7] - pos[0]]
		

		fit_xlist = Append(cut_xlist[0:(pos[1] - pos[0]+1)],cut_xlist[(pos[2] - pos[0]):(pos[3] - pos[0]+1)])
		fit_xlist = Append(fit_xlist,cut_xlist[(pos[4] - pos[0]):(pos[5] - pos[0]+1)])
		fit_xlist = Append(fit_xlist,cut_xlist[(pos[6] - pos[0]):(pos[7] - pos[0]+1)])
		fit_ylist = Append(cut_ylist_f[0:(pos[1] - pos[0]+1)],cut_ylist_f[(pos[2] - pos[0]):(pos[3] - pos[0]+1)])
		fit_ylist = Append(fit_ylist,cut_ylist_f[(pos[4] - pos[0]):(pos[5] - pos[0]+1)])
		fit_ylist = Append(fit_ylist,cut_ylist_f[(pos[6] - pos[0]):(pos[7] - pos[0]+1)])
		chi_ylist = Append(ylist_norm[pos[0]:(pos[1]+1)],ylist_norm[pos[2]:(pos[3]+1)])
		chi_ylist = Append(chi_ylist,ylist_norm[(pos[4]):(pos[5]+1)])
		chi_ylist = Append(chi_ylist,ylist_norm[(pos[6]):(pos[7]+1)]) 
		y7 = (y0 - y1) / (x0-x1) * (fit_xlist - x0) + y0
		'''
		plt.scatter(fit_xlist, fit_ylist - y7, s = 5)
		plt.show()
		'''

		def fGs(x,s1,t1,t2,a,b,s2):
			return fG(x,7155*s1,7155*t1,a)+fG(x,7172*s1,7172*t1,0.24*a)+fG(x,7388*s1,7388*t1,0.19*a)+fG(x,7453*s1,7453*t1,0.31*a)+fG(x,7378*s2,7378*t2,b)+fG(x,7412*s2,7412*t2,0.31*b) + (y0 - y1) / (x0-x1) * (x - x0) + y0

		params,params_covariance=optimize.curve_fit(fGs,fit_xlist,fit_ylist,guess,maxfev=500000,bounds=bound)

		chi2 = np.sum((fGs(fit_xlist,params[0],params[1],params[2],params[3],params[4],params[5]) - fit_ylist)**2)/np.size(fit_xlist)

		print('chi2: %f' %chi2)


		y7 = (y0 - y1) / (x0-x1) * (cut_xlist - x0) + y0
		ysimu = fGs(cut_xlist,params[0],params[1],params[2],params[3],params[4],params[5]) - y7
		y1 = fG(cut_xlist,7155*params[0],7155*params[1],params[3])
		y2 = fG(cut_xlist,7172*params[0],7172*params[1],0.24*params[3])
		y3 = fG(cut_xlist,7388*params[0],7388*params[1],0.19*params[3])
		y4 = fG(cut_xlist,7453*params[0],7453*params[1],0.31*params[3])
		y5 = fG(cut_xlist,7378*params[5],7378*params[2],params[4])
		y6 = fG(cut_xlist,7412*params[5],7412*params[2],0.31*params[4])

		'''
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
		
		plt.plot(cut_xlist, ylist_tempt1, label = 'data', c = 'b')
		plt.plot(cut_xlist, fGs_improve(cut_xlist, params[0], params[1], params[3]), label = 'fit', c = 'r')
		plt.legend(loc = 'upper left')
		plt.show()

		plt.plot(cut_xlist, ylist_tempt2, label = 'data', c = 'b')
		plt.plot(cut_xlist, y5+y6, label = 'fit', c = 'r')
		plt.legend(loc = 'upper left')
		plt.show()
		

		bound_i =  ([sub1,0.98,0],[sup1,1.02,float('inf')])
		guess_i = [sig,1,1]
		params_i, params_covariance_i = optimize.curve_fit(fGs_improve,cut_xlist,ylist_tempt1,guess_i,maxfev=500000,bounds=bound_i)

		
		plt.plot(cut_xlist, ylist_tempt1, label = 'data', c = 'b')
		plt.plot(cut_xlist, fGs_improve(cut_xlist, params_i[0], params_i[1], params_i[2]), label = 'fit', c = 'r')
		plt.legend(loc = 'upper left')
		plt.show()

		params[0] = params_i[0]
		params[1] = params_i[1]
		params[3] = params_i[2]
		
		plt.plot(cut_xlist, cut_ylist_f-y7, label = 'data', c = 'b')
		plt.plot(cut_xlist, fGs_improve(cut_xlist, params[0], params[1], params[3])+y5+y6, label = 'fit_f', c = 'black')
		plt.plot(cut_xlist, ysimu, label = 'fit_i0', c = 'yellow')
		plt.plot(cut_xlist, fGs(cut_xlist,params[0],params[1],params[2],params[3],params[4],params[5]) - y7, label = 'fit_i', c = 'purple')
		plt.legend(loc = 'upper left')
		plt.show()
		'''
 
		FWHM_Fe = params[0] * 2 * 300000 * np.sqrt(2*np.log(2))
		FWHM_Ni = params[5] * 2 * 300000 * np.sqrt(2*np.log(2))
		vshift_Fe = (params[1] - 1) * 300000 
		vshift_Ni = (params[2] - 1) * 300000
		flux_ratio = params[4] * params[5] * 7378 / params[3] / params[0] / 7155
		high_ratio = params[4] / params[3]

		print('FWHM_Fe: %f, FWHM_Ni: %f, vshift_Fe: %f, vshift_Ni: %f, flux_ratio Ni/Fe: %f, high_ratio Ni/Fe: %s' % (FWHM_Fe,FWHM_Ni, vshift_Fe, vshift_Ni, flux_ratio, high_ratio))


		def f1(x):
			return params[4] * params[5] / params[3] / params[0] / 4.9 /np.exp(0.28*1.60217662/1.380694/10**(-4)/x)
		#0.5372471812372228(integrate), 0.5023869679593065(two point mean)
		v, err = itg.quad(f1, 3000, 8000)
		ratio_ave_bef = v / 5000

		print('n_Ni/n_Fe = %f' % (ratio_ave_bef * 1.8))
		'''
		y1 = fG(cut_xlist,params[0],7155*params[1],params[3])
		y2 = fG(cut_xlist,params[0],7172*params[1],0.24*params[3])
		y3 = fG(cut_xlist,params[0],7388*params[1],0.19*params[3])
		y4 = fG(cut_xlist,params[0],7453*params[1],0.31*params[3])
		y5 = fG(cut_xlist,params[5],7378*params[2],params[4])
		y6 = fG(cut_xlist,params[5],7412*params[2],0.31*params[4])
		ysimu = y1 + y2 + y3 + y4 + y5 + y6
		'''

		fig = plt.figure(figsize=(8,6))
		plt.title('%s  +%sd' %(target_name, phase), fontsize=15)
		plt.tick_params(labelsize=15)
		plt.xlabel('Rest Wavelength [$\\rm \\AA$]',fontsize=15)
		plt.ylabel('Scaled Flux',fontsize=15)
		plt.plot(xlist[(pos[0]-100):(pos[7]+100+1)], ylist_norm[(pos[0]-100):(pos[7]+100+1)], color="gray", label="data")
		plt.plot(xlist[(pos[0]-100):(pos[7]+100+1)], ylist_t[(pos[0]-100):(pos[7]+100+1)],color='black',label='smoothed data')
		#plt.plot(cut_xlist, ysimu+y7, color="red",  label="Gaussian fits\n $\\overline{\\chi^2} =$ %f"%chi2)
		plt.plot(cut_xlist, ysimu+y7, color="red",  label="Gaussian fits")
		plt.plot(cut_xlist, y1+y7, color="purple", label="[Fe II]",linestyle='--')
		plt.plot(cut_xlist, y2+y7, color="purple",linestyle='--')
		plt.plot(cut_xlist, y3+y7, color="purple",linestyle='--')
		plt.plot(cut_xlist, y4+y7, color="purple",linestyle='--')
		plt.plot(cut_xlist, y5+y7, color="green", label="[Ni II]",linestyle='--')
		plt.plot(cut_xlist, y6+y7, color="green",linestyle='--')
		plt.plot(cut_xlist[0:(pos[1] - pos[0]+1)], np.ones(pos[1]-pos[0]+1)*min(ylist_t[(pos[0]-100):(pos[7]+100+1)])*0.9, c = 'b', label = 'fit region')
		plt.plot(cut_xlist[(pos[2] - pos[0]):(pos[3] - pos[0]+1)], np.ones(pos[3]-pos[2]+1)*min(ylist_t[(pos[0]-100):(pos[7]+100+1)])*0.9, c = 'b')
		plt.plot(cut_xlist[(pos[4] - pos[0]):(pos[5] - pos[0]+1)], np.ones(pos[5]-pos[4]+1)*min(ylist_t[(pos[0]-100):(pos[7]+100+1)])*0.9, c = 'b')
		plt.plot(cut_xlist[(pos[6] - pos[0]):(pos[7] - pos[0]+1)], np.ones(pos[7]-pos[6]+1)*min(ylist_t[(pos[0]-100):(pos[7]+100+1)])*0.9, c = 'b')
		plt.plot(cut_xlist, y7, color="y",   label="continuum")
		plt.legend()
		plt.show()
		'''
		plt.savefig('./appendix/'+FigureName)
		plt.show(block=False)
		plt.pause(1)
		'''
		plt.close()
		'''
		ymin = np.min(ylist[pos[0]:(pos[5]+1)] - y7)
		plt.title('%s  +%sd' %(target_name, phase))
		plt.xlabel('Rest Wavelength [Å]')
		plt.ylabel('Flux Density')
		plt.yticks([])
		plt.plot(xlist[pos[0]:(pos[5]+1)], ylist[pos[0]:(pos[5]+1)] - y7, color="blue", label="data")
		plt.plot(cut_xlist, ysimu, color="red",  label="best fit")
		plt.plot(cut_xlist, y1, color="purple", label="[Fe II]")
		plt.plot(cut_xlist, y2, color="purple")
		plt.plot(cut_xlist, y3, color="purple")
		plt.plot(cut_xlist, y4, color="purple")
		plt.plot(cut_xlist, y5, color="green", label="[Ni II]")
		plt.plot(cut_xlist, y6, color="green")
		plt.plot(cut_xlist[0:(pos[1] - pos[0]+1)], ymin*np.ones(pos[1]-pos[0]+1), c = 'y', label = 'fit region')
		plt.plot(cut_xlist[(pos[2] - pos[0]):(pos[3] - pos[0]+1)], ymin*np.ones(pos[3]-pos[2]+1), c = 'y')
		plt.plot(cut_xlist[(pos[4] - pos[0]):(pos[5] - pos[0]+1)], ymin*np.ones(pos[5]-pos[4]+1), c = 'y')
		plt.legend(loc='upper left')
		plt.show()
		
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
		q = 3
		if q == 2:
			if (Min2 - Max1) >= 10:
				save_as = '_IMG.dat'
			else:
				save_as = '_IMG.dat'
			with open('result_data0'+save_as,'a') as f:
				f.writelines('%s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' %(target_name, phase, delta15, U_delta15, redshift, E_B_V, vSi, U_vSi, ratio_ave_bef * 1.8, vshift_Fe, vshift_Ni, FWHM_Fe, FWHM_Ni, flux_ratio))
				g = 1
				break
		elif q == 3:
			g = 0
			break
		print(Min1,Max1,Min2,Max2,Min3,Max3,Min4,Max4)
		print(width)
		change_Min1 = int(input('input change_Min1:'))
		change_Max1 = int(input('input change_Max1:'))
		change_Min2 = int(input('input change_Min2:'))
		change_Max2 = int(input('input change_Max2:'))
		change_Min3 = int(input('input change_Min3:'))
		change_Max3 = int(input('input change_Max3:'))
		change_Min4 = int(input('input change_Min4:'))
		change_Max4 = int(input('input change_Max4:'))
		bound =  ([sub1,0.98,0.98,0,0,sub2],[sup1,1.02,1.02,float('inf'),float('inf'),sup2])
		width = int(input('width: '))
		Min1 += change_Min1
		Max1 += change_Max1
		Min2 += change_Min2
		Max2 += change_Max2
		Min3 += change_Min3
		Max3 += change_Max3
		Min4 += change_Min4
		Max4 += change_Max4

	if g == 1:
		#Possion
		cut_xlist = xlist[pos[0]:(pos[7]+1)]
		cut_ylist = ylist_norm[pos[0]:(pos[7]+1)]

		fit_xlist = Append(cut_xlist[0:(pos[1] - pos[0]+1)],cut_xlist[(pos[2] - pos[0]):(pos[3] - pos[0]+1)])
		fit_xlist = Append(fit_xlist,cut_xlist[(pos[4] - pos[0]):(pos[5] - pos[0]+1)])
		fit_xlist = Append(fit_xlist,cut_xlist[(pos[6] - pos[0]):(pos[7] - pos[0]+1)])
		fit_ylist = Append(cut_ylist[0:(pos[1] - pos[0]+1)],cut_ylist[(pos[2] - pos[0]):(pos[3] - pos[0]+1)])
		fit_ylist = Append(fit_ylist,cut_ylist[(pos[4] - pos[0]):(pos[5] - pos[0]+1)])
		fit_ylist = Append(fit_ylist,cut_ylist[(pos[6] - pos[0]):(pos[7] - pos[0]+1)])

		x0 = cut_xlist[0]
		y0 = cut_ylist[0]
		x1 = cut_xlist[pos[7] - pos[0]]
		y1 = cut_ylist[pos[7] - pos[0]]	

		params,params_covariance=optimize.curve_fit(fGs,fit_xlist,fit_ylist,guess,maxfev=500000,bounds=bound)
		'''
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
		'''
		if np.abs(f_ratio/flux_ratio[0]) < 2 and np.abs(f_ratio/flux_ratio[0]) > 0.1 and np.abs(h_ratio/high_ratio[0]) < 3 and np.abs(h_ratio/high_ratio[0]) > 0.1:
		'''	

		Up_FWHM_Fe = FWHM_Fe - params[0] * 2 * 300000 * np.sqrt(2*np.log(2))
		Up_vshift_Fe = vshift_Fe - (params[1] - 1) * 300000
		Up_FWHM_Ni = FWHM_Ni - params[5] * 2 * 300000 * np.sqrt(2*np.log(2))
		Up_vshift_Ni = vshift_Ni - (params[2] - 1) * 300000
		Up_flux_ratio = flux_ratio - params[4] * params[5] * 7378 / params[3] / params[0] / 7155
		Up_high_ratio = high_ratio - params[4]/params[3]
		#edge
		vshift_Fe = [vshift_Fe]
		vshift_Ni = [vshift_Ni]
		FWHM_Fe = [FWHM_Fe]
		FWHM_Ni = [FWHM_Ni]
		flux_ratio = [flux_ratio]
		high_ratio = [high_ratio]

		if initialize:
			edge_size = int(input('edge_size: '))
		else:	
			edge_size = float(ESList[n])
		k=0
		while(k < 1000):
			print(k, end='\r')
			bluet1 = np.random.rand()*edge_size*2 - edge_size + Min1
			if (Min2 - Max1) >= 20:
				redt1 = np.random.rand()*20 - 10 + Max1
				bluet2 = np.random.rand()*20 - 10 + Min2
			else:
				redt1 = Max1
				bluet2 = Min2
			if (Min3 - Max2) >= 10:
				redt2 = np.random.rand()*20 - 10 + Max2
				bluet3 = np.random.rand()*20 - 10 + Min3
			else:
				redt2 = Max2
				bluet3 = Min3
			if (Min4 - Max3) >= 10:
				redt3 = np.random.rand()*20 - 10 + Max3
				bluet4 = np.random.rand()*20 - 10 + Min4
			else:
				redt3 = Max3
				bluet4 = Min4
			redt4 = np.random.rand()*edge_size*2 - edge_size + Max4
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
					indicator = 4
				elif int(xlist[j]) >= bluet3 and indicator == 4:
					pos[4] = j
					indicator = 5
				elif int(xlist[j]) >= redt3 and indicator == 5:
					pos[5] = j
					indicator = 6
				elif int(xlist[i]) >= bluet4 and indicator == 6:
					pos[6] = i
					indicator = 7
				elif int(xlist[i]) >= redt4 and indicator == 7:
					pos[7] = i
					break

			cut_xlist = xlist[pos[0]:(pos[7]+1)]
			cut_ylist = ylist_t[pos[0]:(pos[7]+1)] 

			fit_xlist = Append(cut_xlist[0:(pos[1] - pos[0]+1)],cut_xlist[(pos[2] - pos[0]):(pos[3] - pos[0]+1)])
			fit_xlist = Append(fit_xlist,cut_xlist[(pos[4] - pos[0]):(pos[5] - pos[0]+1)])
			fit_xlist = Append(fit_xlist,cut_xlist[(pos[6] - pos[0]):(pos[7] - pos[0]+1)])
			fit_ylist = Append(cut_ylist[0:(pos[1] - pos[0]+1)],cut_ylist[(pos[2] - pos[0]):(pos[3] - pos[0]+1)])
			fit_ylist = Append(fit_ylist,cut_ylist[(pos[4] - pos[0]):(pos[5] - pos[0]+1)])
			fit_ylist = Append(fit_ylist,cut_ylist[(pos[6] - pos[0]):(pos[7] - pos[0]+1)])


			x0 = cut_xlist[0]
			y0 = cut_ylist[0]
			x1 = cut_xlist[pos[7] - pos[0]]
			y1 = cut_ylist[pos[7] - pos[0]]	

			params,params_covariance=optimize.curve_fit(fGs,fit_xlist,fit_ylist,guess,maxfev=500000,bounds=bound)
			'''
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
			'''
			if np.abs(f_ratio/flux_ratio[0]) < 2 and np.abs(f_ratio/flux_ratio[0]) > 0.1 and np.abs(h_ratio/high_ratio[0]) < 3 and np.abs(h_ratio/high_ratio[0]) > 0.1:
			'''	
			f_ratio = params[4] * params[5] / params[3] / params[0]
			h_ratio = params[4] / params[3]
			FWHM_Nit = params[5] * 2 * 300000 * np.sqrt(2*np.log(2))
			if f_ratio < 2 and f_ratio > 0.01 and h_ratio < 5 and FWHM_Nit < 12500:
				FWHM_Fe.append(params[0] * 2 * 300000 * np.sqrt(2*np.log(2)))
				FWHM_Ni.append(params[5] * 2 * 300000 * np.sqrt(2*np.log(2)))
				vshift_Fe.append((params[1] - 1) * 300000) 
				vshift_Ni.append((params[2] - 1) * 300000)
				flux_ratio.append(params[4] * params[5] *7378 / params[3] / params[0] / 7155)
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
		print('Up_FWHM_Fe: %s, Up_vshift_Fe: %s, Up_FWHM_Ni: %s, Up_vshift_Ni: %s, Up_flux_ratio: %s, Up_high_ratio: %s' %(Up_FWHM_Fe, Up_vshift_Fe, Up_FWHM_Ni, Up_vshift_Ni, Up_flux_ratio, Up_high_ratio))
		print('U_FWHM_Fe: %s, U_vshift_Fe: %s, U_FWHM_Ni: %s, U_vshift_Ni: %s, U_flux_ratio: %s, U_high_ratio: %s' %(U_FWHM_Fe, U_vshift_Fe, U_FWHM_Ni, U_vshift_Ni, U_flux_ratio, U_high_ratio))
		with open('Uncertenty'+save_as,'a') as f:
			f.writelines('%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' %(target_name, phase, U_vshift_Fe, U_vshift_Ni, U_FWHM_Fe, U_FWHM_Ni, U_flux_ratio, Min1, Max1, Min2, Max2, Min3, Max3, Min4, Max4, width, edge_size))
		g = 0
