import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import scipy.signal as signal
from extinction import fitzpatrick99
from scipy.optimize import curve_fit
from scipy import stats

def Append(l1, l2):
	l3 = []
	for item in l1:
		l3.append(item)
	for item in l2:
		l3.append(item)
	return l3

def rdata(filename, start):
	with open(filename,'r') as f:
		for i in range (start):
			line = f.readline()
		nametemp = []
		ptemp = []
		dtemp = []
		Udtemp = []
		rtemp = []
		redshift = []
		ebv = []
		vSitemp = []
		UvSitemp = []
		vFetemp = []
		vNitemp = []
		wFetemp = []
		wNitemp   = []
		ratio_flux = []
		subc = []
		a = line.split()
		while line:
			a = line.split()
			nametemp.append(a[0])
			ptemp.append(int(a[1]))
			dtemp.append(float(a[2]))
			Udtemp.append(float(a[3]))
			redshift.append(float(a[4]))
			ebv.append(float(a[5]))
			vSitemp.append(float(a[6]))
			UvSitemp.append(float(a[7]))
			rtemp.append(float(a[8]))
			vFetemp.append(float(a[9]))
			vNitemp.append(float(a[10]))
			wFetemp.append(float(a[11]))
			wNitemp.append(float(a[12]))
			ratio_flux.append(float(a[13]))
			subc.append(a[14])
			line = f.readline()
	number = len(ptemp)
	return nametemp, ptemp, dtemp, Udtemp, vSitemp, UvSitemp, rtemp, vFetemp, vNitemp, wFetemp, wNitemp, ratio_flux, subc, number

def rU(filename, start):
	with open(filename,'r') as f:
		for i in range (start):
			line = f.readline()
		UvFetemp = []
		UvNitemp = []
		UwFetemp = []
		UwNitemp   = []
		Uratio_flux = []
		edge_blue = []
		edge_red = []
		width = []
		a = line.split()
		while line:
			a = line.split()
			UvFetemp.append(float(a[2]))
			UvNitemp.append(float(a[3]))
			UwFetemp.append(float(a[4]))
			UwNitemp.append(float(a[5]))
			Uratio_flux.append(float(a[6]))
			edge_blue.append(float(a[7]))
			edge_red.append(float(a[14]))
			width.append(int(a[15]))
			line = f.readline()
	return UvFetemp, UvNitemp, UwFetemp, UwNitemp, Uratio_flux, edge_blue, edge_red, width

def f1(x,a,b):
	return a*x + b

def color(a):
	if a == 0:
		return 'r'
	elif a == 1:
		return 'b'
	else:
		return 'gray'

def insert(xlist, pos, element):
	N = np.size(xlist)
	xlist.append(xlist[N-1])
	for i in range(N-1, pos, -1):
		xlist[i] = xlist[i-1]
	xlist[pos] = element

labelsize = 15

name = ['SN1990N', 'SN1991T','SN1993Z','SN1994ae','SN1995D','SN1996X','SN1998aq','SN1998bu','SN1999aa','SN2002bo','SN2002dj','SN2002er',
'SN2003cg','SN2003du','SN2003gs','SN2003hv','SN2003kf','SN2004eo','SN2005cf','SN2006X','SN2007af','SN2007le','SN2008Q',
'SN2009ig','SN2009le','SN2010ev','SN2010gp','SN2011at','SN2011by','SN2011ek','SN2011fe','SN2011im','SN2011iv',
'SN2012cg','SN2012fr','SN2012hr','SN2012ht','SN2013aa','SN2013cs','SN2013dy','SN2013gy','SN2014J',
'SN2015F','ASASSN-14jg']
'''
v_Fe = ['b', 'y', 'r', 'y', 'y', ''
]Silverman madea 17 21 
'''
v_Fe = ['b', 'y', 'y', 'y', 'y', 'y', 'y', 'b', 'gray', 'r', 'y', 'y',
'y', 'b', 'r', 'b', 'y', 'b', 'gray', 'r', 'r', 'r', 'b',
'y', 'y', 'y','y','y', 'b', 'y', 'b', 'y', 'y',
'b', 'r', 'r', 'r', 'b', 'r', 'b', 'b', 'r',
'b', 'r']
v_SiII = [9.38, 9.8, 0, 11.1, 10.1, 11.3, 10.7, 10.8, 10.5, 13.2, 13.4, 11.7,
10.9,10.4,11.4,11.3,11.1,10.7,10.1,16.1,10.8,12.9,11.09,
13.0,0,14.98,0,0,10.35,0,10.4,0,10.4,
10.0,11.93,11.5,11.0,10.2,12.5,10.3,10.7,12.1,
10.1,10.20]

Uv_SiII = [0.15, 0.2, 0, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.2, 0.2, 0.3,
0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.6, 0.10,
0.3, 0,0.02,0,0,0.14, 0,0.2, 0,0.2,
0.2, 0.1, 0.2, 0.2,0.2, 0.2, 0.2, 0.2, 0.2,
0.2, 0.2]

N = len(name)
name = ['SN1990N', 'SN1991T','SN1993Z','SN1994ae','SN1995D','SN1996X','SN1998aq','SN1998bu','SN1999aa','SN2002bo','SN2002dj','SN2002er',
'SN2003cg','SN2003du','SN2003gs','SN2003hv','SN2003kf','SN2004eo','SN2005cf','SN2006X','SN2007af','SN2007le','SN2008Q',
'SN2009ig','SN2009le','SN2010ev','SN2010gp','SN2011at','SN2011by','SN2011ek','SN2011fe','SN2011im','SN2011iv',
'SN2012cg','SN2012fr','SN2012hr','SN2012ht','SN2013aa','SN2013cs','SN2013dy','SN2013gy','SN2014J',
'SN2015F','ASASSN-14jg']
Phase = [280,316,233,368,284.7,246,241.5,280,256,227,275,216,
385,272,201,323,397,228,319,277,301,304,201,
405,324,272,279,349,310,423,311,314,318,
279,261,283,433,399,303,333,276,351,
280.1,323]
delta =  [0.95,0.94,0.00,0.89,1.01,1.30,1.02,1.06,0.81,1.15,1.13,1.32,
1.17,1.00,1.93,1.45,0.93,1.40,1.07,1.26,1.16,1.02,1.25,
0.90,0.00,0.00,0.00,0.00,1.14,0.00,1.18,0.00,0.00, 
0.83,0.85,1.04,0.00,0.95,0.81,0.92,1.20,0.98,
1.18,0.92]
Udelta = [0.05,0.03,0.00,0.05,0.03,0.05,0.03,0.04,0.02,0.03,0.03,0.03,
0.04,0.02,0.07,0.07,0.03,0.03,0.06,0.05,0.03,0.05,0.08,
0.05,0.06,0.06,0.06,0.06,0.03,0.00,0.03,0.07,0.05,
0.03,0.05,0.01,0.00,0.01,0.18,0.01,0.03,0.02,
0.02,0.01]

ratio = [0.023,0.031,0.033,0.050,0.008,0.053,0.037,0.059,0.053,0.051,0.051,0.083,
0.049,0.031,0.054,0.087,0.032,0.055,0.029,0.065,0.035,0.032,0.081,
0.028,0.036,0.044,0.033,0.059,0.039,0.020,0.041,0.047,0.051,
0.025,0.030,0.021,0.009,0.027,0.029,0.025,0.053,0.033,
0.050,0.039]
tUratio = [[0.005,0.011,0.007,0.009,0.006,0.020,0.011,0.008,0.009,0.010,0.010,0.019,
0.008,0.009,0.008,0.012,0.007,0.009,0.005,0.008,0.006,0.005,0.011,
0.005,0.006,0.007,0.005,0.010,0.006,0.005,0.006,0.013,0.018,
0.005,0.006,0.006,0.004,0.005,0.006,0.005,0.009,0.007,
0.006,0.007],
[0.004,0.009,0.006,0.008,0.006,0.018,0.010,0.008,0.008,0.009,0.008,0.016,
0.007,0.008,0.007,0.010,0.006,0.008,0.006,0.007,0.005,0.005,0.010,
0.005,0.005,0.006,0.005,0.009,0.006,0.006,0.005,0.011,0.015,
0.005,0.005,0.006,0.004,0.006,0.005,0.005,0.008,0.006,
0.006,0.006]]
Uratio = []
for i in range(N):
	Uratio.append([[tUratio[0][i]],[tUratio[1][i]]])

'''
cm = plt.cm.get_cmap('Blues')
plt.xlabel('The decline-rate parameter [magnitude]')
plt.ylabel('$\\rm M_{Ni}/M_{Fe}$')
for i in range(N):
	if delta[i] != 0.00:
		plt.errorbar(delta[i], ratio[i], xerr = Udelta[i],yerr = Uratio[i], capsize = 3, linestyle = '-', marker = 'o', c = 'b')
plt.scatter(0.6, 0.05, alpha = 0)
plt.legend(loc = 'upper left', frameon=False)
plt.show()
plt.xlabel('The decline-rate parameter [magnitude]')
plt.ylabel('$\\rm M_{Ni}/M_{Fe}$')
for i in range(N):
	if delta[i] != 0.00:
		plt.scatter(delta[i], ratio[i], c=phase[i], s = 20, vmin = 150, vmax = 430, cmap = cm)
plt.colorbar(label = 'phase since maximum')
plt.show()

plt.xlabel('The velocity of Si II at maximum [km/s]')
plt.ylabel('$\\rm M_{Ni}/M_{Fe}$')
for i in range(N):
	if v_SiII[i] != 0:
		plt.errorbar(v_SiII[i],ratio[i],xerr = Uv_SiII[i],yerr = Uratio[i], capsize = 3, marker = 'o', c = 'b')
plt.show()
plt.xlabel('The velocity of Si II at maximum [km/s]')
plt.ylabel('$\\rm M_{Ni}/M_{Fe}$')
for i in range(N):
	if v_SiII[i] != 0:
		plt.scatter(v_SiII[i], ratio[i], c=phase[i], s = 20, vmin = 150, vmax = 430, cmap = cm)
plt.colorbar(label = 'phase since maximum')
plt.show()
'''
name1, phase1, delta1, Udelta1, vSi1, UvSi1, ratio1, vFe1, vNi1, wFe1, wNi1, r_flux1, subc, number1 = rdata('result_data0_IMG.dat',1)
name1.append('tail')
UvFe1, UvNi1, UwFe1, UwNi1, U_r_flux1, edge_blue, edge_red, width = rU('Uncertenty_IMG.dat',1)

name2, phase2, delta2, Udelta2, vSi2, UvSi2, ratio2, vFe2, vNi2, wFe2, wNi2, r_flux2, subc2, number2 = rdata('result_data0_pr.dat',1)
name2.append('tail')
UvFe2, UvNi2, UwFe2, UwNi2, U_r_flux2, edge_blue2, edge_red2, width2 = rU('Uncertenty_pr.dat',1)


print(number1)
print(number2)

def Fe56(t, Ni, Co, lambda_Ni, lambda_Co):
	return Co + 1 - Co*np.exp(-lambda_Co*t) - lambda_Co*Ni/(lambda_Co-lambda_Ni)*np.exp(-lambda_Ni*t) + lambda_Ni*Ni/(lambda_Co-lambda_Ni)*np.exp(-lambda_Co*t)


def get_quene(array):
	qlist = []
	plist = []
	size = np.size(array)
	if size < 2:
		qlist.append(0)
		plist.append(0)
		return qlist, plist
	array_sort = np.sort(array)
	for i in range(size):
		for j in range(size):
			if array[i] == array_sort[j]:
				qlist.append(j)
				break
	for i in range(size):
		n=1
		for j in range(i+1,size):
			if qlist[i] == qlist[j]:
				qlist[j] += n
				n += 1
	for i in range(size):
		for j in range(size):
			if i == qlist[j]:
				plist.append(j)
				break
	return qlist, plist

qlist, plist = get_quene(delta1)

fig, ax = plt.subplots()
plt.tick_params(labelsize=15)
ax.set_xlabel('Wavelength [$\\rm \\AA$]', fontsize=15)
ax.set_ylabel('Scaled Flux + constant', fontsize=15)
ax.set_yticks([])
spectra_x = []
spectra_y = []
n = 0
for i in range(number1):
	sepctrum_name = name1[i] + '/' + name1[i] + '_' + str(phase1[i]) +'.dat'
	with open('paper/' + sepctrum_name,'r') as f:
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
	length = len(xlist)

	'''
	if phase < 300 and phase > 250 and n<5 and name1[i]!='SN1986G':
		Min1 = 4000
		Max1 = 9000
		pos = [0,np.size(xlist)-1]
		indicator = 0
		for j in range(length):
			if int(xlist[j]) >= Min1 and indicator == 0:
				pos[0] = j
				indicator = 1
			elif int(xlist[j]) >= Max1 and indicator == 1:
				pos[1] = j
				break
		plt.plot(xlist[pos[0]:(pos[1])], ylist[pos[0]:(pos[1])]/np.max(ylist[pos[0]:(pos[1])]) + 0.5*n, c = 'black')
		plt.text(9100,ylist[pos[1]]+0.5*n, name1[i] + ', +%s d' %phase1[i])
		plt.scatter(10000,0,alpha=0)
		n += 1
	'''
	if name1[i] == 'SN2021hpr' or name1[i] == 'SN2021wuf' or name1[i] == 'SN2019np' or name1[i] == 'SN2019ein':
		Min1 = 4000
		Max1 = 10000
		pos = [0,np.size(xlist)-1]
		indicator = 0
		for j in range(length):
			if int(xlist[j]) >= Min1 and indicator == 0:
				pos[0] = j
				indicator = 1
			elif int(xlist[j]) >= Max1 and indicator == 1:
				pos[1] = j
				break
		xlist_smoothed = xlist[pos[0]:pos[1]]
		ylist_smoothed = signal.savgol_filter(ylist[pos[0]:pos[1]], 61, 1)
		scale_max = np.max(ylist_smoothed)
		plt.plot(xlist_smoothed, ylist_smoothed/scale_max*5-2*n, c = 'black')
		plt.text(8000, ylist_smoothed[-25]/scale_max*5-2*n+0.4, name1[i] + ', +%s d' %phase1[i], size=10)
		n += 1
	Min1 = 6600
	Max1 = 8000
	xlist = xlist / (1+redshift)
	ylist = ylist*np.power(10,fitzpatrick99(xlist,R_V*E_B_V,R_V)/2.5)
	

	pos = [0,np.size(xlist)-1]
	indicator = 0
	for j in range(length):
		if int(xlist[j]) >= Min1 and indicator == 0:
			pos[0] = j
			indicator = 1
		elif int(xlist[j]) >= Max1 and indicator == 1:
			pos[1] = j
			break
	ylist = ylist/np.max(signal.savgol_filter(ylist[pos[0]:(pos[1]+1)],51,1))
	spectra_x.append(xlist[pos[0]:(pos[1]+1)])
	spectra_y.append(ylist[pos[0]:(pos[1]+1)])
	#spectra_y.append(ylist[pos[0]:(pos[1]+1)]+0.02*qlist[i])

plt.axvline(4659, linestyle='--', c='gray')
plt.axvline(7155, linestyle='--', c='gray')
plt.axvline(7378, linestyle='--', c='gray')
plt.show()

np.random.seed(399991)
fig, ax = plt.subplots()
plt.tick_params(labelsize=25)
ax.set_xlabel('Wavelength [$\\rm \\AA$]',fontsize=25)
ax.set_ylabel('Scaled Flux',fontsize=25)
i = 0
k=[]
k2=[]
shift_factor = 0.8
while(i < number1):
	if name1[i] == 'SN2014J':
		#color = (np.random.rand(),np.random.rand(),np.random.rand())
		n = i
		plt.title('SN 2014J', fontsize=25)
		spectra_y_t = signal.savgol_filter(spectra_y[i], width[i], 1)
		

		start = np.argmax(np.array(spectra_x[i])>=edge_blue[i])
		end = np.argmax(np.array(spectra_x[i])>=edge_red[i])
		continuum_x = [spectra_x[i][start], spectra_x[i][end]]
		continuum_y = [spectra_y_t[start], spectra_y_t[end]]
		
		for j in range(number2):
			if name2[j] == 'SN2014J':
				m = j
				break
		start2 = np.argmax(np.array(spectra_x[i])>=edge_blue2[m])
		end2 = np.argmax(np.array(spectra_x[i])>=edge_red2[m])
		continuum_x2 = [spectra_x[i][start2], spectra_x[i][end2]]
		continuum_y2 = [spectra_y_t[start2], spectra_y_t[end2]]
		k.append((continuum_y[1] - continuum_y[0])/(continuum_x[1] - continuum_x[0]))
		k2.append((continuum_y2[1] - continuum_y2[0])/(continuum_x2[1] - continuum_x2[0]))
		plt.plot(continuum_x, continuum_y, c='b', label = 'pseudo-continuum', linestyle='--')
		plt.plot(continuum_x2, continuum_y2, c='r', label = 'pseudo-continuum', linestyle='--')

		plt.plot(spectra_x[i], spectra_y_t, label='+%d d'%phase1[i])

		size_x = np.size(spectra_x[i])
		while(name1[n+1] == 'SN2014J'):
			n += 1
			m += 1
			spectra_y_t = signal.savgol_filter(spectra_y[n], width[n], 1)
			plt.plot(spectra_x[n], spectra_y_t, label='+%d d'%phase1[n])
			start = np.argmax(np.array(spectra_x[n])>=edge_blue[n])
			end = np.argmax(np.array(spectra_x[n])>=edge_red[n])
			continuum_x = [spectra_x[n][start], spectra_x[n][end]]
			continuum_y = [spectra_y_t[start], spectra_y_t[end]]
			start2 = np.argmax(np.array(spectra_x[n])>=edge_blue2[m])
			end2 = np.argmax(np.array(spectra_x[n])>=edge_red2[m])
			continuum_x2 = [spectra_x[n][start2], spectra_x[n][end2]]
			continuum_y2 = [spectra_y_t[start2], spectra_y_t[end2]]
			k.append((continuum_y[1] - continuum_y[0])/(continuum_x[1] - continuum_x[0]))
			k2.append((continuum_y2[1] - continuum_y2[0])/(continuum_x2[1] - continuum_x2[0]))
			plt.plot(continuum_x, continuum_y, c='b', linestyle='--')
			plt.plot(continuum_x2, continuum_y2, c='r', linestyle='--')
			size_x = np.size(spectra_x[n])
		i += n-i
	'''
	if name1[i] == 'SN2012fr':
		n = i
		plt.plot(spectra_x[i], spectra_y[i])
		size_x = np.size(spectra_x[i])
		plt.text(spectra_x[i][size_x-1]+50, spectra_y[i][size_x-1], name1[i] + ', +%s d' %phase1[i])
		while(name1[n+1] == 'SN2012fr'):
			n += 1
			plt.plot(spectra_x[n], spectra_y[n] + (n-i)*0.025)
			size_x = np.size(spectra_x[n])
			plt.text(spectra_x[n][size_x-1]+50, spectra_y[n][size_x-1] + (n-i)*0.025, name1[n] + ', +%s d' %phase1[n])
		i += n-i
	if name1[i] == 'SN2014J':
		n = i
		plt.plot(spectra_x[i], spectra_y[i]+0.02)
		size_x = np.size(spectra_x[i])
		plt.text(spectra_x[i][size_x-1]+50, spectra_y[i][size_x-1]+0.02, name1[i] + ', +%s d' %phase1[i])
		while(name1[n+1] == 'SN2014J'):
			n += 1
			plt.plot(spectra_x[n], spectra_y[n]+0.02)
			size_x = np.size(spectra_x[n])
			plt.text(spectra_x[n][size_x-1]+50, spectra_y[n][size_x-1]+0.02, name1[n] + ', +%s d' %phase1[n])
		i += n-i
	'''
	i += 1


collection = collections.BrokenBarHCollection.span_where(np.linspace(6800,6987,100),ymin=0,ymax=0.6,where=np.ones(100)>0,facecolor='gray',alpha=0.5)
ax.add_collection(collection)
plt.text(6750, 0.61, 'excess flux', fontsize=20)
plt.legend(fontsize=15)
plt.show()

print('k: ', k)
print('k2: ', k2)



'''
cm = plt.cm.get_cmap('viridis')
for i in range(13):
	plt.xlabel('Rest Wavelength [Å]')
	plt.ylabel('Scaled Flux + constant')
	plt.yticks([])
	plt.scatter(spectra_x[plist[i*4+2]], spectra_y[plist[i*4+2]], c=delta1[plist[i*4+2]]*np.ones(np.size(spectra_x[plist[i*4+2]])), vmin = 0.8, vmax = 1.5, cmap = cm, s=1)
	size_x = np.size(spectra_x[plist[i*4+2]])
	plt.text(spectra_x[plist[i*4+2]][size_x-1]+50, spectra_y[plist[i*4+2]][size_x-1], name1[plist[i*4+2]] + ', +%s d' %phase1[plist[i*4+2]])
plt.colorbar(label = '$\\Delta m_{15}(B)$')
plt.plot(9500,np.min(spectra_y[plist[2]]),alpha=0)
plt.show()

ylist = ylist*np.max(signal.savgol_filter(ylist[pos[0]:(pos[1]+1)],51,1))/np.max(ylist[pos[0]:(pos[1]+1)])

for j in range(4):
	for i in range(13*j, 13*(j+1)):
		plt.xlabel('Rest Wavelength [Å]')
		plt.ylabel('Scaled Flux + constant')
		plt.yticks([])
		plt.scatter(spectra_x[plist[i]], spectra_y[plist[i]]+0.01*3*i, c=delta1[plist[i]]*np.ones(np.size(spectra_x[plist[i]])), vmin = 0.8, vmax = 1.5, cmap = cm, s=1)
		size_x = np.size(spectra_x[plist[i]])
		plt.text(spectra_x[plist[i]][size_x-1]+50, spectra_y[plist[i]][size_x-1]+0.01*3*i, name1[plist[i]] + ', +%s d' %phase1[plist[i]])
	plt.colorbar(label = '$\\Delta m_{15}(B)$')
	plt.plot(9500,np.min(spectra_y[plist[13*j]]+0.01*3*i),alpha=0)
	plt.show()
'''
marker = ['.','o','v','^','<','>','s','p','x','d','*','h','H','+','x','D','d','_','.','o','v','^','<','>','s','p','*','h','H','+','x','D','x','d','_']

sub_shape = {"N":"o", "91T":"v", "91bg":"^"}

integrate = 0.5372471812372228
mean = 0.5023869679593065

def Fe56(t, Ni, Co, lambda_Ni, lambda_Co):
	return Co + Ni - Co*np.exp(-lambda_Co*t) - lambda_Co*Ni/(lambda_Co-lambda_Ni)*np.exp(-lambda_Ni*t) + lambda_Ni*Ni/(lambda_Co-lambda_Ni)*np.exp(-lambda_Co*t)

lambda_Ni = 1/8.77
lambda_Co = 1/111

u1 = 0.4
U_ratio1 = []
U_ratio2 = []
for i in range(number1):
	ratio1[i] = ratio1[i]*58/56*mean/integrate*Fe56(phase1[i]+18, 1, 0, lambda_Ni, lambda_Co)
	u4 = U_r_flux1[i]/r_flux1[i]
	UvFe1[i] += 200
	UvNi1[i] += 200
	U_ratio1_t = np.sqrt(u1**2 + u4**2)*ratio1[i]
	U_ratio1.append(U_ratio1_t)

for i in range(number2):
	ratio2[i] = ratio2[i]*58/56*mean/integrate*Fe56(phase2[i]+18, 1, 0, lambda_Ni, lambda_Co)
	u4 = U_r_flux2[i]/r_flux2[i]
	UvFe2[i] += 200
	UvNi2[i] += 200
	U_ratio2_t = np.sqrt(u1**2 + u4**2)*ratio2[i]
	U_ratio2.append(U_ratio2_t)

np.random.seed(399991)

for i in range(number1+1):
	if name1[i] == "SN2014J":
		begin1 = i
		while(1):
			i += 1
			if name1[i] != "SN2014J":
				end1 = i
				break
		break

for i in range(number2+1):
	if name2[i] == "SN2014J":
		begin2 = i
		while(1):
			i += 1
			if name2[i] != "SN2014J":
				end2 = i
				break
		break
print(begin1, end1, begin2, end2)
fig = plt.figure(figsize=(8,10))
ax = fig.subplots(4,1,sharex=True)

ax[0].tick_params(labelsize=20)
ax[0].plot(phase1[begin1:end1], r_flux1[begin1:end1], c='b')
ax[0].errorbar(phase1[begin1:end1], r_flux1[begin1:end1], yerr = U_r_flux1[begin1:end1], c = 'b', capsize = 3, linestyle = '-', marker = 'o', label='bluer blue endpoint')
ax[0].plot(phase2[begin2:end2], r_flux2[begin2:end2], c='r')
ax[0].errorbar(phase2[begin2:end2], r_flux2[begin2:end2], yerr = U_r_flux2[begin2:end2], c = 'r', capsize = 3, linestyle = '-', marker = 'o', label='reder blue endpoint')
ax[0].set_ylabel('$F_{7378}/F_{7155}$',fontsize=20)
ax[0].legend(fontsize=15)

ax[1].tick_params(labelsize=20)
ax[1].set_ylabel('[Fe II] Vel.\n [km/s]', fontsize=20)
ax[1].plot(phase1[begin1:end1], vFe1[begin1:end1], c='b')
ax[1].errorbar(phase1[begin1:end1], vFe1[begin1:end1], yerr = UvFe1[begin1:end1], c = 'b', capsize = 3, linestyle = '-', marker = 'o')
ax[1].plot(phase2[begin2:end2], vFe2[begin2:end2], c='r')
ax[1].errorbar(phase2[begin2:end2], vFe2[begin2:end2], yerr = UvFe2[begin2:end2], c = 'r', capsize = 3, linestyle = '-', marker = 'o')

ax[2].tick_params(labelsize=20)
ax[2].set_ylabel('[Ni II] Vel.\n [km/s]', fontsize=20)
ax[2].plot(phase1[begin1:end1], vNi1[begin1:end1], c='b')
ax[2].errorbar(phase1[begin1:end1], vNi1[begin1:end1], yerr = UvNi1[begin1:end1], c = 'b', capsize = 3, linestyle = '-', marker = 'o')
ax[2].plot(phase2[begin2:end2], vNi2[begin2:end2], c='r')
ax[2].errorbar(phase2[begin2:end2], vNi2[begin2:end2], yerr = UvNi2[begin2:end2], c = 'r', capsize = 3, linestyle = '-', marker = 'o')

ax[3].tick_params(labelsize=20)
ax[3].set_ylabel('[Ni II] FWHM\n [km/s]', fontsize=20)
ax[3].plot(phase1[begin1:end1], wNi1[begin1:end1], c='b')
ax[3].errorbar(phase1[begin1:end1], wNi1[begin1:end1], yerr = UwNi1[begin1:end1], c = 'b', capsize = 3, linestyle = '-', marker = 'o')
ax[3].plot(phase2[begin2:end2], wNi2[begin2:end2], c='r')
ax[3].errorbar(phase2[begin2:end2], wNi2[begin2:end2], yerr = UwNi2[begin2:end2], c = 'r', capsize = 3, linestyle = '-', marker = 'o')
ax[3].set_xlabel('Phase [Days Since Peak Brightness]',fontsize=20)
#fig.savefig('figures/cpr_para.pdf', bbox_inches='tight')
plt.show()



np.random.seed(399991)
plt.xlabel('Phase [Days Since Peak Brightness]')
plt.ylabel('Velocity of [Fe II] [km/s]')
plt.plot(np.linspace(150,450,100),np.zeros(100),linestyle='--',c = 'black')
line = []
head = 0
n = 0
for i in range(1,number1+1):
	if name1[head] != name1[i]:
		color = (np.random.rand(),np.random.rand(),np.random.rand())
		plt.errorbar(phase1[head:i],vFe1[head:i],yerr = UvFe1[head:i],label = '%s'%name1[head], c=color, capsize = 3, linestyle = '-', marker = marker[n])
		line_t, = plt.plot(phase1[head:i],vFe1[head:i],label = '%s'%name1[head], c=color, linestyle = '-', marker = marker[n])
		line.append(line_t)
		head = i
		n += 1
l1 = plt.legend(handles=line[0:int(n/2)], loc = 'upper left')
plt.legend(handles=line[int(n/2):], loc = 'upper right')
plt.gca().add_artist(l1)
plt.show()

np.random.seed(399991)
plt.xlabel('Phase [Days Since Peak Brightness]')
plt.ylabel('[Ni II] velocity[km/s]')
plt.plot(np.linspace(150,450,100),np.zeros(100),linestyle='--',c = 'black')
line = []
head = 0
n = 0
for i in range(1,number1+1):
	if name1[head] != name1[i]:
		color = (np.random.rand(),np.random.rand(),np.random.rand())
		plt.errorbar(phase1[head:i],vNi1[head:i],yerr = UvNi1[head:i],label = '%s'%name1[head], c=color, capsize = 3, linestyle = '-', marker = marker[n])
		line_t, = plt.plot(phase1[head:i],vNi1[head:i],label = '%s'%name1[head], c=color, linestyle = '-', marker = marker[n])
		line.append(line_t)
		head = i
		n += 1
l1 = plt.legend(handles=line[0:int(n/2)], loc = 'upper left')
plt.legend(handles=line[int(n/2):], loc = 'upper right')
plt.gca().add_artist(l1)
plt.show()

vNebular1 = (np.array(vNi1)+np.array(vFe1))/2
UvNebular1 = np.sqrt(np.array(UvNi1)**2+np.array(UvFe1)**2)/2
np.random.seed(399991)
plt.tick_params(labelsize=12)
plt.xlabel('Phase [Days Since Peak Brightness]',fontsize=12)
plt.ylabel('Nebular Velocity[km s$^{-1}$]',fontsize=12)
plt.plot(np.linspace(150,450,100),np.zeros(100),linestyle='--',c = 'black')
line = []
head = 0
n = 0
for i in range(1,number1+1):
	if name1[head] != name1[i]:
		color = (np.random.rand(),np.random.rand(),np.random.rand())
		plt.errorbar(phase1[head:i],vNebular1[head:i],yerr = UvNebular1[head:i],label = '%s'%name1[head], c=color, capsize = 3, linestyle = '-', marker = marker[n])
		line_t, = plt.plot(phase1[head:i],vNebular1[head:i],label = '%s'%name1[head], c=color, linestyle = '-', marker = marker[n])
		line.append(line_t)
		head = i
		n += 1
l1 = plt.legend(handles=line[0:int(n/2)], loc = 'upper left')
plt.legend(handles=line[int(n/2):], loc = 'upper right')
plt.gca().add_artist(l1)
plt.show()

def Fe56(t, Ni, Co, lambda_Ni, lambda_Co):
	return Co + Ni - Co*np.exp(-lambda_Co*t) - lambda_Co*Ni/(lambda_Co-lambda_Ni)*np.exp(-lambda_Ni*t) + lambda_Ni*Ni/(lambda_Co-lambda_Ni)*np.exp(-lambda_Co*t)



day_list = np.linspace(110,450,2)

double_sup = 0.061
double_sub = 0.015

ModelNameList = ['n1','n10', 'n100', 'n100h', 'n100l', 'n100_z0.01', 'n150', 'n1600', 'n1600c', 'n20', 'n200', 'n3', 'n300c', 'n40', 'n5']
Ni_58_List = []
Ni_56_List = []
Co_56_list = []
for item in ModelNameList:
	with open('models/ddt_2013_' + item + '_abundances.dat','r') as f:
		line = f.readline()
		while line:
			if line == '\n':
				line = f.readline()
				continue
			a = line.split()
			if a[0] == 'ni56':
				Ni_56_List.append(float(a[1]))
			if a[0] == 'co56':
				Co_56_list.append(float(a[1]))
			if a[0] == 'ni58':
				Ni_58_List.append(float(a[1]))
			line = f.readline()
for i in range(np.size(ModelNameList)):
	if ModelNameList[i] == 'n3':
		Ni_56 = Ni_56_List[i]
		Co_56 = Co_56_list[i]
		Ni_58 = Ni_58_List[i]
		ratio_n3 = Ni_58 / Ni_56
	if ModelNameList[i] == 'n20':
		Ni_56 = Ni_56_List[i]
		Co_56 = Co_56_list[i]
		Ni_58 = Ni_58_List[i]
		ratio_n20 = 0.104

np.random.seed(399991)
fig, ax = plt.subplots()
ax.tick_params(labelsize=labelsize)
ax.set_xlabel('Phase [Days Since Peak Brightness]',fontsize=labelsize)
ax.set_ylabel('$\\rm M_{Ni}/M_{Fe}, t \\rightarrow \\infty$',fontsize=labelsize)

ax.fill_between(day_list, np.ones_like(day_list)*double_sub, np.ones_like(day_list)*double_sup, alpha=0.5, color = 'gray')
ax.fill_between(day_list, np.ones_like(day_list)*ratio_n3, np.ones_like(day_list)*ratio_n20, alpha=0.5, color = 'yellow')
print(ratio_n3)
print(ratio_n20)
line = []
head = 0
n = 0
for i in range(1,number1+1):
	if name1[head] != name1[i]:
		color = (np.random.rand(),np.random.rand(),np.random.rand())
		ax.errorbar(phase1[head:i],ratio1[head:i],yerr = np.array(U_ratio1[head:i]), capsize = 3, linestyle = '-', c = color, marker = marker[n])
		line_t, = ax.plot(phase1[head:i],ratio1[head:i],label = '%s'%name1[head], linestyle = '-', c = color,marker = marker[n])
		line.append(line_t)
		if name1[head] == 'SN2002bo':
			ax.text(phase1[head]+5, ratio1[head], '02bo', color=color)
		if name1[head] == 'SN2015F':
			ax.text(phase1[head]+5, ratio1[head], '15F', color=color)
		head = i
		n += 1
l1 = ax.legend(handles=line[0:int(n/2)], loc = 'upper left')
ax.legend(handles=line[int(n/2):], loc = 'upper right')
plt.gca().add_artist(l1)
ax.text(110,0.02,'sub-M$_{Ch}$ Double Det.',fontsize=15)
ax.text(110,0.07,'M$_{Ch}$ Del. Det.',fontsize=15)
plt.show()


j0 = 0

green_n = 0
purple_n = 0
for i in range(N):
	for j in range(j0, number1, 1):
		if name[i] == name1[j] and Phase[i] == phase1[j]:
			if Phase[i] < 300:
				if green_n == 0:
					plt.scatter(ratio1[j], ratio[i], marker = 'o', c = 'green', label = '< +300 d')
					green_n = 1
				plt.errorbar(ratio1[j], ratio[i], xerr = U_ratio1[j], yerr = Uratio[i], capsize = 3, linestyle = '-', marker = 'o', c = 'green')
			else:
				if purple_n == 0:
					plt.scatter(ratio1[j], ratio[i], marker = 'o', c = 'purple', label = '> +300 d')
					purple_n = 1
				plt.errorbar(ratio1[j], ratio[i], xerr = U_ratio1[j], yerr = Uratio[i], capsize = 3, linestyle = '-', marker = 'o', c = 'purple')
				
			j0 = j+1
			break
plt.tick_params(labelsize=25)
plt.xlabel('This work',fontsize=25)
plt.ylabel('Fl$\\rm \\"o$rs et al. (2020)',fontsize=25)
plt.plot(np.linspace(0,np.max(ratio),10), np.linspace(0,np.max(ratio),10), c = 'grey', linestyle = '--',label='y=x')
plt.text(0.00, 0.095,'$\\rm M_{Ni}/M_{Fe}, t \\rightarrow \\infty$',fontsize=25)
#plt.text(0.154,0.045,'15F',fontsize=15)
#plt.plot([0.154,0.1474],[0.045,0.049],c='black')
plt.text(0.09,0.027,'14J',fontsize=15)
plt.plot([0.09,0.086],[0.027,0.0317],c='black')
plt.text(0.0277,0.056,'03gs',fontsize=15)
plt.plot([0.033,0.0265],[0.056,0.055],c='black')
plt.text(0.0,0.046,'99aa',fontsize=15)
plt.plot([0.0164,0.022],[0.048,0.0524],c='black')
plt.legend(fontsize=25)
plt.show()

exit()
'''
jlist=[]
head = 0
for i in range(1,number1):
	if name1[head] != name1[i]:
		jlist.append(i-1)
		head = i
jlist.append(number1-1)
'''
jlist = []
head = 0
for i in range(1, number1+1):
	if name1[head] != name1[i]:
		jlist.append(i-1)
		head = i

no_C = 0
phase_C =[]
no_subC = 0
phase_subC = []
no_critical = 0
for i in range(np.size(jlist)):
	if ratio1[jlist[i]] < 0.06:
		no_subC += 1
		phase_subC.append(phase1[jlist[i]])
	elif ratio1[jlist[i]] > 0.064:
		no_C += 1
		phase_C.append(phase1[jlist[i]])
	else:
		no_critical += 1

print('C: %d' %no_C)
print(phase_C)
print('subC: %d' %no_subC)
print(phase_subC)
print('critical: %d' %no_critical)

jlist=[]
head = 0
for i in range(1,number1+1):
	if name1[head] != name1[i]:
		phase_dif_tempt = np.abs(np.array(phase1[head:i]) - 300)
		qlist, plist = get_quene(phase_dif_tempt)
		jlist.append(head+plist[0])
		head = i


print()
g_vSi = []
Ug_vSi = []
g_ratio = []
Ug_ratio = []
g_delta = []
Ug_delta = []
g_vN = []
Ug_vN = []
g_subc = []
b_vSi = []
Ub_vSi = []
b_ratio = []
Ub_ratio = []
b_delta = []
Ub_delta = []
b_vN = []
Ub_vN = []
b_subc = []
r_vSi = []
Ur_vSi = []
r_ratio = []
Ur_ratio = []
r_delta = []
Ur_delta = []
r_vN = []
Ur_vN = []
r_subc = []
for i in range(np.size(jlist)):
	vN_t = (vFe1[jlist[i]]+vNi1[jlist[i]])/2
	UvN_t = np.sqrt(UvFe1[jlist[i]]**2 + UvNi1[jlist[i]]**2)/2
	if abs(vN_t)<abs(UvN_t) and ratio1[jlist[i]] < 0.13:
		print(name1[jlist[i]])
		g_vSi.append(vSi1[jlist[i]])
		Ug_vSi.append(UvSi1[jlist[i]])
		g_ratio.append(ratio1[jlist[i]])
		Ug_ratio.append(U_ratio1[jlist[i]])
		g_delta.append(delta1[jlist[i]])
		Ug_delta.append(Udelta1[jlist[i]])
		g_vN.append((vFe1[jlist[i]]+vNi1[jlist[i]])/2)
		Ug_vN.append(np.sqrt(UvFe1[jlist[i]]**2+UvNi1[jlist[i]]**2)/2)
		g_subc.append(subc[jlist[i]])
	elif vN_t < 0 and ratio1[jlist[i]] < 0.13:
		b_vSi.append(vSi1[jlist[i]])
		Ub_vSi.append(UvSi1[jlist[i]])
		b_ratio.append(ratio1[jlist[i]])
		Ub_ratio.append(U_ratio1[jlist[i]])
		b_delta.append(delta1[jlist[i]])
		Ub_delta.append(Udelta1[jlist[i]])
		b_vN.append((vFe1[jlist[i]]+vNi1[jlist[i]])/2)
		Ub_vN.append(np.sqrt(UvFe1[jlist[i]]**2+UvNi1[jlist[i]]**2)/2)
		b_subc.append(subc[jlist[i]])
	elif vN_t > 0 and ratio1[jlist[i]] < 0.13:
		r_vSi.append(vSi1[jlist[i]])
		Ur_vSi.append(UvSi1[jlist[i]])
		r_ratio.append(ratio1[jlist[i]])
		Ur_ratio.append(U_ratio1[jlist[i]])
		r_delta.append(delta1[jlist[i]])
		Ur_delta.append(Udelta1[jlist[i]])
		r_vN.append((vFe1[jlist[i]]+vNi1[jlist[i]])/2)
		Ur_vN.append(np.sqrt(UvFe1[jlist[i]]**2+UvNi1[jlist[i]]**2)/2)
		r_subc.append(subc[jlist[i]])
		if vSi1[jlist[i]] < 12:
			print(name1[jlist[i]], vSi1[jlist[i]], (vFe1[jlist[i]]+vNi1[jlist[i]])/2)
	elif ratio1[jlist[i]] > 0.13:
		print(name1[jlist[i]])
	else:
		print(name1[jlist[i]])

print('blue: %d' %np.size(b_vSi))
print('red: %d' %np.size(r_vSi))
print('gray: %d' %np.size(g_vSi))



def f_line(x,a,b):
	return a*x + b

params_b, params_covariance_b = curve_fit(f_line, b_vSi, b_ratio, [1,0])
params_r, params_covariance_r = curve_fit(f_line, r_vSi, r_ratio, [1,0])

def kendalltau_err(x, y, xerr, yerr):
	N = 20001
	tau = []
	p = []
	for i in range(N):
		x_t = x + np.random.normal(scale=xerr)
		y_t = y + np.random.normal(scale=yerr)
		tau_t, p_t = stats.kendalltau(x_t, y_t)
		tau.append(tau_t)
		p.append(p_t)
	return np.percentile(tau, [16, 50, 84]), np.percentile(p, [16, 50, 84])

def pearson_err(x,y,xerr,yerr):
	N = 20001
	r = []
	p = []
	for i in range(N):
		x_t = x + np.random.normal(scale=xerr)
		y_t = y + np.random.normal(scale=yerr)
		res = stats.pearsonr(x_t, y_t)
		r_t = res[0]
		p_t = res[1]
		r.append(r_t)
		p.append(p_t)
	return np.percentile(r, [16, 50, 84]), np.percentile(p, [16, 50, 84])
'''
tau_result_b, p_result_b = kendalltau_err(b_vSi,b_ratio, Ub_vSi, Ub_ratio)
print(tau_result_b)
print(p_result_b)
tau_b, p_value_b = stats.kendalltau(b_vSi,b_ratio)
print('tau_b, p_value_b: ',tau_b, p_value_b)

tau_result_r, p_result_r = kendalltau_err(r_vSi,r_ratio, Ur_vSi, Ur_ratio)
print(tau_result_r)
print(p_result_r)
tau_r, p_value_r = stats.kendalltau(r_vSi,r_ratio)
print('tau_r, p_value_r: ',tau_r, p_value_r)

pearson_r_blue, pearson_p_blue = pearson_err(b_vSi,b_ratio, Ub_vSi, Ub_ratio)
pearson_r_red,  pearson_p_red  = pearson_err(r_vSi,r_ratio, Ur_vSi, Ur_ratio)
print(stats.pearsonr(b_vSi, b_ratio))
print(pearson_r_blue)
print(pearson_p_blue)
print(stats.pearsonr(r_vSi, r_ratio))
print(pearson_r_red)
print(pearson_p_red)
'''
# definitions for the axes
left, width = 0.1, 0.55
bottom, height = 0.1, 0.55
spacing = 0.015


rect_scatter = [left, bottom + 0.2 + spacing, width, height]
rect_histx = [left, bottom, width, 0.2]
rect_histy = [left + width + spacing, bottom + 0.2 + spacing, 0.2, height]

# start with a square Figure
fig = plt.figure(figsize=(6, 6))
ax = fig.add_axes(rect_scatter)
ax.tick_params(labelsize=13)
ax.tick_params(axis="x", labelbottom=False)
ax.set_ylabel('$\\rm M_{Ni}/M_{Fe}, t \\rightarrow \\infty$',fontsize=13)
ax.set_xlim([9,17])
ax.set_ylim([0,0.14])
ax_histx = fig.add_axes(rect_histx, sharex=ax)
ax_histx.tick_params(labelsize=13)
ax_histy = fig.add_axes(rect_histy, sharey=ax)
ax_histy.tick_params(labelsize=13)

# no labels
ax_histx.set_xlabel('Si II Velocity Near Peak Brightness [$\\rm 10^3\\ km\\ s^{-1}$]',fontsize=12)
ax_histx.set_ylabel('Number',fontsize=13)
ax_histy.tick_params(axis="y", labelleft=False)
ax_histy.set_xlabel('Number',fontsize=13)

# the scatter plot:
for i in range(np.size(g_vSi)):
	if i == 0:
		ax.scatter(g_vSi[i],g_ratio[i], c = 'gray', marker = 'o', label = 'Zero')
	ax.errorbar(g_vSi[i],g_ratio[i],xerr = Ug_vSi[i],yerr = Ug_ratio[i], c = 'gray', capsize = 3, linestyle = '-', marker = 'o')
for i in range(np.size(b_vSi)):
	if i == 0:
		ax.scatter(b_vSi[i],b_ratio[i], c = 'b', marker = 'o', label = 'Blue-shifted')
	ax.errorbar(b_vSi[i],b_ratio[i],xerr = Ub_vSi[i],yerr = Ub_ratio[i], c = 'b', capsize = 3, linestyle = '-', marker = 'o')
for i in range(np.size(r_vSi)):
	if i == 0:
		ax.scatter(r_vSi[i],r_ratio[i], c = 'r', marker = 'o', label = 'Red-shifted')
	ax.errorbar(r_vSi[i],r_ratio[i],xerr = Ur_vSi[i],yerr = Ur_ratio[i], c = 'r', capsize = 3, linestyle = '-', marker = 'o')
ax.plot(np.linspace(9,13,10), f_line(np.linspace(9,13,10), params_b[0], params_b[1]), c = 'b', linestyle = '--')
ax.plot(np.linspace(9,17,10), f_line(np.linspace(9,17,10), params_r[0], params_r[1]), c = 'r', linestyle = '--')
ax.text(9.61,0.0182,'14jg',c='r')
ax.text(12.66,0.0793,'21wuf',c='b')
ax.text(11.18,0.0793,'21hpr',c='r')
ax.text(15.08,0.0768,'17fgc',c='r')
ax.text(16.21,0.0721,'06X',c='r')

# now determine nice limits by hand:
x_bins = [9,10,11,12,13,14,15,16,17]
y_bins = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,1.1]
ax_histx.hist(r_vSi, bins=x_bins, color='r', label = 'Red-shifted')
ax_histx.hist(b_vSi, bins=x_bins, edgecolor='b', histtype='step', label = 'Blue-shifted')
ax_histx.legend(fontsize=13)
ax_histy.set_xlim([0,8])
ax_histy.hist(r_ratio, bins=y_bins, orientation='horizontal', color='r', label = 'Red-shifted')
ax_histy.hist(b_ratio, bins=y_bins, orientation='horizontal', edgecolor='b', histtype='step', label = 'Blue-shifted')
ax_histy.legend(fontsize=13)

ax.legend(fontsize=12)
plt.show()

plt.tick_params(labelsize=15)
plt.xlabel('$Si II Velocity Near Peak Brightness [$\\rm 10^3\\ km\\ s^{-1}$]',fontsize=15)
plt.ylabel('$Nebular Velocity [$\\rm \\ km\\ s^{-1}$]',fontsize=15)
for i in range(np.size(g_vSi)):
	plt.errorbar(g_vSi[i],g_vN[i],xerr = Ug_vSi[i],yerr = Ug_vN[i], c = 'gray', capsize = 3, linestyle = '-', marker = 'o')
for i in range(np.size(b_vSi)):
	plt.errorbar(b_vSi[i],b_vN[i],xerr = Ub_vSi[i],yerr = Ub_vN[i], c = 'b', capsize = 3, linestyle = '-', marker = 'o')
for i in range(np.size(r_vSi)):
	plt.errorbar(r_vSi[i],r_vN[i],xerr = Ur_vSi[i],yerr = Ur_vN[i], c = 'r', capsize = 3, linestyle = '-', marker = 'o')
plt.show()
'''
plt.xlabel('The velocity of Si II at maximum [$\\rm 10^3\\ km\\ s^{-1}$]')
plt.ylabel('$\\rm {\\Delta}m_{15}(B)$ [magnitude]')
for i in range(np.size(g_vSi)):
	plt.errorbar(g_vSi[i],g_delta[i],xerr = Ug_vSi[i],yerr = Ug_delta[i], c = 'gray', capsize = 3, linestyle = '-', marker = 'o')
for i in range(np.size(b_vSi)):
	plt.errorbar(b_vSi[i],b_delta[i],xerr = Ub_vSi[i],yerr = Ub_delta[i], c = 'b', capsize = 3, linestyle = '-', marker = 'o')
for i in range(np.size(r_vSi)):
	plt.errorbar(r_vSi[i],r_delta[i],xerr = Ur_vSi[i],yerr = Ur_delta[i], c = 'r', capsize = 3, linestyle = '-', marker = 'o')
plt.show()
'''
plt.tick_params(labelsize=20)
plt.xlabel('Nebular Velocity [$\\rm \\ km\\ s^{-1}$]',fontsize=20)
plt.ylabel('$\\rm {\\Delta}m_{15}(B)$ [magnitude]',fontsize=20)
for i in range(np.size(g_vSi)):
	if i == 0:
		plt.scatter(g_vN[i],g_delta[i], c = 'gray', marker = 'o', label = 'Zero')
	plt.errorbar(g_vN[i],g_delta[i],xerr = Ug_vN[i],yerr = Ug_delta[i], c = 'gray', capsize = 3, linestyle = '-', marker = 'o')
for i in range(np.size(b_vSi)):
	if i == 0:
		plt.scatter(b_vN[i],b_delta[i], c = 'blue', marker = 'o', label = 'Blue-shifted')
	plt.errorbar(b_vN[i],b_delta[i],xerr = Ub_vN[i],yerr = Ub_delta[i], c = 'b', capsize = 3, linestyle = '-', marker = 'o')
for i in range(np.size(r_vSi)):
	if i == 0:
		plt.scatter(r_vN[i],r_delta[i], c = 'red', marker = 'o', label = 'Red-shifted')
	plt.errorbar(r_vN[i],r_delta[i],xerr = Ur_vN[i],yerr = Ur_delta[i], c = 'r', capsize = 3, linestyle = '-', marker = 'o')
plt.legend(fontsize=15)
plt.show()
exit()

delta_tau = np.array(Append(r_delta,b_delta))
delta_tau = np.array(Append(delta_tau,g_delta))
Udelta_tau = np.array(Append(Ur_delta, Ub_delta))
Udelta_tau = np.array(Append(Udelta_tau, Ug_delta))
ratio_tau = np.array(Append(r_ratio,b_ratio))
ratio_tau = np.array(Append(ratio_tau,g_ratio))
Uratio_tau = np.array(Append(Ur_ratio, Ub_ratio))
Uratio_tau = np.array(Append(Uratio_tau, Ug_ratio))
delete_pos = []
for i in range(np.size(delta_tau)):
	if delta_tau[i] > 1.7:
		delete_pos.append(i)
delta_tau = np.delete(delta_tau, delete_pos)
Udelta_tau = np.delete(Udelta_tau, delete_pos)
ratio_tau = np.delete(ratio_tau, delete_pos)
Uratio_tau = np.delete(Uratio_tau, delete_pos)
'''
pearson_r_15, pearson_p_15 = pearson_err(delta_tau,ratio_tau, Udelta_tau, Uratio_tau)
print(pearson_r_15, pearson_p_15)
print(stats.pearsonr(delta_tau, ratio_tau))
plt.scatter(delta_tau, ratio_tau)
plt.show()

tau_result_15, p_result_15 = kendalltau_err(delta_tau,ratio_tau, Udelta_tau, Uratio_tau)
print(tau_result_15)
print(p_result_15)
tau, p_value = stats.kendalltau(delta_tau, ratio_tau)
print('tau, p_value: ',tau, p_value)
'''
fig, ax = plt.subplots(figsize=(8,6))
ax.fill_between(np.linspace(0.8,2.0,2), np.ones(2)*double_sub, np.ones(2)*double_sup, alpha=0.5, color = 'gray')
ax.fill_between(np.linspace(0.8,2.0,2), np.ones(2)*ratio_n3, np.ones(2)*ratio_n20, alpha=0.5, color = 'yellow')
plt.tick_params(labelsize=20)
plt.xlabel('$\\rm {\\Delta}m_{15}(B)$ [magnitude]',fontsize=20)
plt.ylabel('$\\rm M_{Ni}/M_{Fe}, t \\rightarrow \\infty$',fontsize=20)
for i in range(np.size(g_vSi)):
	if i == 0:
		ax.scatter(g_delta[i],g_ratio[i], c = 'gray', marker = 'o', label = 'Zero')
	plt.errorbar(g_delta[i],g_ratio[i],xerr = Ug_delta[i],yerr = Ug_ratio[i], c = 'gray', capsize = 3, linestyle = '-', marker = 'o')
for i in range(np.size(b_delta)):
	if i == 0:
		ax.scatter(b_delta[i],b_ratio[i], c = 'blue', marker = 'o', label = 'Blue-shifted')
	plt.errorbar(b_delta[i],b_ratio[i],xerr = Ub_delta[i],yerr = Ub_ratio[i], c = 'b', capsize = 3, linestyle = '-', marker = 'o')
for i in range(np.size(r_vSi)):
	if i == 0:
		ax.scatter(r_delta[i],r_ratio[i], c = 'red', marker = 'o', label = 'Red-shifted')
	plt.errorbar(r_delta[i],r_ratio[i],xerr = Ur_delta[i],yerr = Ur_ratio[i], c = 'r', capsize = 3, linestyle = '-', marker = 'o')
ax.text(1.5,0.015,'sub-M$_{Ch}$ Double Det.',fontsize=15)
ax.text(1.5,0.07,'M$_{Ch}$ Del. Det.',fontsize=15)
ax.text(1.75,0.04,'86G', c='b',fontsize=15)
ax.text(1.96,0.0325,'03gs',c='r',fontsize=15)
ax.text(0.75,0.01,'99aa',c='gray',fontsize=15)
plt.legend(fontsize=20)
plt.show()

for i in range(np.size(jlist)):
	plt.scatter(delta1[jlist[i]], vSi1[jlist[i]])
plt.show()
'''
vSit = []
UvSit = []
ratiot = []
Uratiot = []
deltat = []
Udeltat = []
jlist = []
for i in range(np.size(v_SiII)):
	if v_SiII[i] != 0 and v_Fe[i] != 'y' and delta[i] != 0.00:
		jlist.append(i)
		vSit.append(v_SiII[i])
		ratiot.append(ratio[i])
		UvSit.append(Uv_SiII[i])
		Uratiot.append(Uratio[i])
		deltat.append(delta[i])
		Udeltat.append(Udelta[i])
plt.xlabel('The velocity of Si II at maximum [$10^3$km/s]')
plt.ylabel('$M_{Ni}/M_{Fe}$')
for i in range(np.size(jlist)):
	plt.errorbar(vSit[i],ratiot[i],xerr = UvSit[i],yerr = Uratiot[i], c = v_Fe[jlist[i]], capsize = 3, linestyle = '-', marker = 'o')
	if v_Fe[jlist[i]] == 'r':
		print(name[jlist[i]])
		print(vSit[i])
		print(ratiot[i])
plt.show()
plt.xlabel('The decline-rate parameter [magnitude]')
plt.ylabel('$M_{Ni}/M_{Fe}$')
for i in range(np.size(jlist)):
	plt.errorbar(deltat[i],ratiot[i],xerr = Udeltat[i],yerr = Uratiot[i], c = v_Fe[jlist[i]], capsize = 3, linestyle = '-', marker = 'o')
	if v_Fe[jlist[i]] == 'r':
		print(name[jlist[i]])
		print(deltat[i])
		print(ratiot[i])
plt.show()


plt.xlabel('Improved Multi-Gaussian')
plt.ylabel('Multi-Gaussian')
jlist=[]
head = 0
n = 0
for i in range(1,number1+1):
	if name1[head] != name1[i]:
		if head + 1 != i:
			for j in range(head, i-1, 1):
				d_to_300_first = np.abs(phase1[j]-300)
				d_to_300_second = np.abs(phase1[j+1]-300)
				if d_to_300_first >= d_to_300_second:
					if j + 2 == i:
						head = j + 1
						break
				else:
					head = j
					break
		plt.errorbar(ratio1[head], ratio2[head], xerr = 0.5*ratio1[head], yerr = 0.5*ratio2[head], capsize = 3, linestyle = '-', marker = 'o', c = 'b')
		jlist.append(head)
		head = i
		n += 1
plt.plot(np.linspace(0,np.max(ratio1)*1.2,10), np.linspace(0,np.max(ratio1)*1.2,10), c = 'grey', linestyle = '-')
plt.scatter(0.05, 0.05, label = '$M_{Ni}/M_{Fe}$', alpha = 0)
plt.legend(loc = 'upper left', frameon=False)
plt.show()
print(np.size(jlist))

plt.xlabel('Improved Multi-Gaussian')
plt.ylabel('Multi-Gaussian')
head = 0
n = 0
for i in range(1,number1):
	if name1[head] != name1[i]:
		if head + 1 != i:
			for j in range(head, i-1, 1):
				d_to_300_first = np.abs(phase1[j]-300)
				d_to_300_second = np.abs(phase1[j+1]-300)
				if d_to_300_first >= d_to_300_second:
					if j + 2 == i:
						head = j + 1
						break
				else:
					head = j
					break
		plt.errorbar(r_flux1[head], r_flux2[head], xerr = U_r_flux1[i], yerr = U_r_flux2[i], capsize = 3, linestyle = '-', marker = 'o', c = 'b')
		head = i
		n += 1
plt.plot(np.linspace(0,np.max(r_flux2)*1.2,10), np.linspace(0,np.max(r_flux2)*1.2,10), c = 'grey', linestyle = '-')
plt.scatter(0.05, 0.05, label = '$Flux_{Ni}/Flux_{Fe}$', alpha = 0)
plt.legend(loc = 'upper left', frameon=False)
plt.show()

plt.xlabel('Improved Multi-Gaussian')
plt.ylabel('Multi-Gaussian')
head = 0
n = 0
for i in range(1,number1):
	if name1[head] != name1[i]:
		if head + 1 != i:
			for j in range(head, i-1, 1):
				d_to_300_first = np.abs(phase1[j]-300)
				d_to_300_second = np.abs(phase1[j+1]-300)
				if d_to_300_first >= d_to_300_second:
					if j + 2 == i:
						head = j + 1
						break
				else:
					head = j
					break
		if n == 0:
			plt.errorbar(vNi1[head], vNi2[head], xerr = UvNi1[head], yerr = UvNi2[head], capsize = 3, linestyle = '-', marker = 'o', c = 'b', label = '[Ni II]')
			plt.errorbar(vFe1[head], vFe2[head], xerr = UvFe1[head], yerr = UvFe2[head], capsize = 3, linestyle = '-', marker = 'o', c = 'r', label = '[Fe II]')
		else:
			plt.errorbar(vNi1[head], vNi2[head], xerr = UvNi1[head], yerr = UvNi2[head], capsize = 3, linestyle = '-', marker = 'o', c = 'b')
			plt.errorbar(vFe1[head], vFe2[head], xerr = UvFe1[head], yerr = UvFe2[head], capsize = 3, linestyle = '-', marker = 'o', c = 'r')
		head = i
		n += 1
plt.plot(np.linspace(np.min(vNi2),np.max(vNi2)*1.2,10), np.linspace(np.min(vNi2),np.max(vNi2)*1.2,10), c = 'grey', linestyle = '-')
plt.scatter(0.05, 0.05, label = 'Velocity [10$^3$ km s$^{-1}$]', alpha = 0)
plt.legend(loc = 'upper left', frameon=False)
plt.show()

plt.xlabel('Improved Multi-Gaussian')
plt.ylabel('Multi-Gaussian')
head = 0
n = 0
for i in range(1,number1):
	if name1[head] != name1[i]:
		if head + 1 != i:
			for j in range(head, i-1, 1):
				d_to_300_first = np.abs(phase1[j]-300)
				d_to_300_second = np.abs(phase1[j+1]-300)
				if d_to_300_first >= d_to_300_second:
					if j + 2 == i:
						head = j + 1
						break
				else:
					head = j
					break
		if n == 0:
			plt.errorbar(wNi1[head], wNi2[head], xerr = UwNi1[head], yerr = UwNi2[head], capsize = 3, linestyle = '-', marker = 'o', c = 'b', label = '[Ni II]')
			plt.errorbar(wFe1[head], wFe2[head], xerr = UwFe1[head], yerr = UwFe2[head], capsize = 3, linestyle = '-', marker = 'o', c = 'r', label = '[Fe II]')
		else:
			plt.errorbar(wNi1[head], wNi2[head], xerr = UwNi1[head], yerr = UwNi2[head], capsize = 3, linestyle = '-', marker = 'o', c = 'b')
			plt.errorbar(wFe1[head], wFe2[head], xerr = UwFe1[head], yerr = UwFe2[head], capsize = 3, linestyle = '-', marker = 'o', c = 'r')
		head = i
		n += 1
plt.plot(np.linspace(np.min(wNi2),np.max(wNi2)*1.02,10), np.linspace(np.min(wNi2),np.max(wNi2)*1.02,10), c = 'grey', linestyle = '-')
plt.scatter(5000, 5000, label = 'FWHM [10$^3$ km s$^{-1}$]', alpha = 0)
plt.legend(loc = 'upper left', frameon=False)
plt.show()
'''