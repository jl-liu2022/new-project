import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import scipy.signal as signal
from extinction import fitzpatrick99
from scipy.optimize import curve_fit




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

			line = f.readline()
	return UvFetemp, UvNitemp, UwFetemp, UwNitemp, Uratio_flux, edge_blue, edge_red

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

ratio = [0.023,0.031,0.033,0.050,0.008,0.053,0.037,0.059,0.055,0.051,0.051,0.083,
0.049,0.031,0.054,0.087,0.032,0.055,0.029,0.065,0.035,0.032,0.081,
0.028,0.036,0.044,0.033,0.059,0.039,0.020,0.041,0.047,0.051,
0.025,0.028,0.021,0.009,0.026,0.022,0.025,0.057,0.026,
0.050,0.039]
tUratio = [[0.005,0.011,0.007,0.009,0.006,0.020,0.011,0.008,0.012,0.010,0.010,0.019,
0.008,0.009,0.008,0.012,0.007,0.009,0.005,0.008,0.006,0.005,0.011,
0.005,0.006,0.007,0.005,0.010,0.006,0.005,0.006,0.013,0.018,
0.005,0.005,0.006,0.004,0.005,0.010,0.005,0.010,0.005,
0.006,0.007],
[0.004,0.009,0.006,0.008,0.006,0.018,0.010,0.008,0.010,0.009,0.008,0.016,
0.007,0.008,0.007,0.010,0.006,0.008,0.006,0.007,0.005,0.005,0.010,
0.005,0.005,0.006,0.005,0.009,0.006,0.006,0.005,0.011,0.015,
0.005,0.005,0.006,0.004,0.005,0.008,0.005,0.008,0.006,
0.006,0.006]]
Uratio = []
for i in range(N):
	Uratio.append([[tUratio[0][i]],[tUratio[1][i]]])
phase = [280,316,233,368,284.7,246,241.5,280.4,282.6,227.7,275,216,
385,272,201,323,397.3,228,319.6,277.6,301,304.7,201.1,
405,324,272,279,349,310,423,311,314,318,
279,290,283,433,344,300,333,280,282,
280,323]
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
print(name1)
name1.append('tail')
UvFe1, UvNi1, UwFe1, UwNi1, U_r_flux1, edge_blue, edge_red = rU('Uncertenty_IMG.dat',1)

print(number1)

def get_quene(array):
	size = np.size(array)
	if size < 2:
		print('size < 2')
		return 0
	array_sort = np.sort(array)
	qlist = []
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
	plist = []
	for i in range(size):
		for j in range(size):
			if i == qlist[j]:
				plist.append(j)
				break
	return qlist, plist

qlist, plist = get_quene(delta1)

spectra_x = []
spectra_y = []
for i in range(number1):
	sepctrum_name = name1[i] + '/' + name1[i] + '_' + str(phase1[i]) +'.dat'
	with open('/Users/pro/python/spectra_data/paper/' + sepctrum_name,'r') as f:
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
	ylist = ylist/np.max(signal.savgol_filter(ylist[pos[0]:(pos[1]+1)],51,1))*0.05 
	spectra_x.append(xlist[pos[0]:(pos[1]+1)])
	spectra_y.append(ylist[pos[0]:(pos[1]+1)]+0.01*qlist[i])

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
fig, ax = plt.subplots()
ax.set_xlabel('波长 [Å]')
ax.set_ylabel('等比缩放的流量 + 常数')
ax.set_yticks([])
i = 0
while(i < number1):
	if name1[i] == 'SN2011fe':
		n = i
		plt.plot(spectra_x[i], spectra_y[i] - 0.1)
		start = np.argmax(np.array(spectra_x[i])>=edge_blue[i])
		print(start)
		end = np.argmax(np.array(spectra_x[i])>=edge_red[i])
		continuum_x = [spectra_x[i][start], spectra_x[i][end]]
		continuum_y = [spectra_y[i][start]-0.1, spectra_y[i][end]-0.1]
		plt.plot(continuum_x, continuum_y, c='gray', label = '连续谱')
		size_x = np.size(spectra_x[i])
		plt.text(spectra_x[i][size_x-1]+50, spectra_y[i][size_x-1] - 0.1, name1[i] + ', +%s d' %phase1[i])
		while(name1[n+1] == 'SN2011fe'):
			n += 1
			plt.plot(spectra_x[n], spectra_y[n] - 0.1)
			start = np.argmax(np.array(spectra_x[n])>=edge_blue[n])
			end = np.argmax(np.array(spectra_x[n])>=edge_red[n])
			continuum_x = [spectra_x[n][start], spectra_x[n][end]]
			continuum_y = [spectra_y[n][start]-0.1, spectra_y[n][end]-0.1]
			plt.plot(continuum_x, continuum_y, c='gray')
			size_x = np.size(spectra_x[n])
			plt.text(spectra_x[n][size_x-1]+50, spectra_y[n][size_x-1] - 0.1, name1[n] + ', +%s d' %phase1[n])
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

plt.plot(8700,0.3,alpha=0)
collection = collections.BrokenBarHCollection.span_where(np.linspace(6800,7000,100),ymin=0,ymax=0.4,where=np.ones(100)>0,facecolor='gray',alpha=0.5)
ax.add_collection(collection)
plt.legend(loc='upper right')
plt.show()

cm = plt.cm.get_cmap('viridis')
for i in range(13):
	plt.xlabel('静止波长 [Å]')
	plt.ylabel('等比缩放的流量 + 常数')
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
		plt.xlabel('精致波长 [Å]')
		plt.ylabel('等比缩放的流量 + 常数')
		plt.yticks([])
		plt.scatter(spectra_x[plist[i]], spectra_y[plist[i]]+0.01*3*i, c=delta1[plist[i]]*np.ones(np.size(spectra_x[plist[i]])), vmin = 0.8, vmax = 1.5, cmap = cm, s=1)
		size_x = np.size(spectra_x[plist[i]])
		plt.text(spectra_x[plist[i]][size_x-1]+50, spectra_y[plist[i]][size_x-1]+0.01*3*i, name1[plist[i]] + ', +%s d' %phase1[plist[i]])
	plt.colorbar(label = '$\\Delta m_{15}(B)$')
	plt.plot(9500,np.min(spectra_y[plist[13*j]]+0.01*3*i),alpha=0)
	plt.show()

marker = ['.','o','v','^','<','>','s','p','x','d','*','h','H','+','x','D','d','_','.','o','v','^','<','>','s','p','*','h','H','+','x','D','x','d','_']

sub_shape = {"N":"o", "91T":"v", "91bg":"^"}

integrate = 0.5372471812372228
mean = 0.5023869679593065

u1 = 0.4
U_ratio1 = []
for i in range(number1):
	ratio1[i] = ratio1[i]*58/56*mean/integrate
	u4 = U_r_flux1[i]/r_flux1[i]
	UvFe1[i] += 200
	UvNi1[i] += 200
	U_ratio1_t = np.sqrt(u1**2 + u4**2)*ratio1[i]
	U_ratio1.append(U_ratio1_t)

np.random.seed(399991)
plt.xlabel('阶段 (光极大之后天数)')
plt.ylabel('[Fe II]速度 [km/s]')
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
plt.xlabel('阶段 (光极大之后天数)')
plt.ylabel('[Ni II]速度 [km/s]')
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
plt.xlabel('阶段 (光极大之后天数)')
plt.ylabel('星云速度 [km/s]')
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

np.random.seed(399991)
fig, ax = plt.subplots()
ax.set_xlabel('阶段 (光极大之后天数)')
ax.set_ylabel('$\\rm M_{Ni}/M_{Fe}$')
ax.scatter(150,0, alpha = 0)
ax.scatter(430,0, alpha = 0)
line = []
head = 0
n = 0
for i in range(1,number1+1):
	if name1[head] != name1[i]:
		color = (np.random.rand(),np.random.rand(),np.random.rand())
		ax.errorbar(phase1[head:i],ratio1[head:i],yerr = np.array(U_ratio1[head:i]), capsize = 3, linestyle = '-', c = color,marker = marker[n])
		line_t, = ax.plot(phase1[head:i],ratio1[head:i],label = '%s'%name1[head], linestyle = '-', c = color,marker = marker[n])
		line.append(line_t)
		head = i
		n += 1
l1 = ax.legend(handles=line[0:int(n/2)], loc = 'upper left')
ax.legend(handles=line[int(n/2):], loc = 'upper right')
plt.gca().add_artist(l1)
collection = collections.BrokenBarHCollection.span_where(np.linspace(150,450,100),ymin=0.016,ymax=0.06,where=np.ones(100)>0,facecolor='gray',alpha=0.5)
ax.add_collection(collection)
collection = collections.BrokenBarHCollection.span_where(np.linspace(150,450,100),ymin=0.064,ymax=0.104,where=np.ones(100)>0,facecolor='yellow',alpha=0.5)
ax.add_collection(collection)
ax.text(150,0.02,'sub-M$_{ch}$ Double Det.')
ax.text(150,0.07,'M$_{ch}$ Del. Det.')
plt.show()



j0 = 0
for i in range(N):
	for j in range(j0, number1, 1):
		if name[i] == name1[j]:
			plt.errorbar(ratio1[j], ratio[i], xerr = U_ratio1[j], yerr = Uratio[i], capsize = 3, linestyle = '-', marker = 'o', c = 'b')
			j0 = j+1
			break
plt.xlabel('This work')
plt.ylabel('Flor')
plt.plot(np.linspace(0,np.max(ratio),10), np.linspace(0,np.max(ratio),10), c = 'grey', linestyle = '--')
plt.scatter(0.05, 0.05, label = '$\\rm M_{Ni}/M_{Fe}$', alpha = 0)
plt.legend(loc = 'upper left', frameon=False)
plt.show()

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
		jlist.append(head)
		head = i
		n += 1
print(np.size(jlist))

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

no_C = 0
no_subC = 0
no_critical = 0
for i in range(np.size(jlist)):
	if ratio1[jlist[i]] < 0.06:
		no_subC += 1
	elif ratio1[jlist[i]] > 0.064:
		no_C += 1
	else:
		no_critical += 1

print('C: %d' %no_C)
print('subC: %d' %no_subC)
print('critical: %d' %no_critical)

def f_line(x,a,b):
	return a*x + b

params_b, params_covariance_b = curve_fit(f_line, b_vSi, b_ratio, [1,0])
params_r, params_covariance_r = curve_fit(f_line, r_vSi, r_ratio, [1,0])

fig, ax = plt.subplots()
plt.xlabel('Si II velocity at max. [$\\rm 10^3\\ km\\ s^{-1}$]')
plt.ylabel('$\\rm M_{Ni}/M_{Fe}$')
for i in range(np.size(g_vSi)):
	plt.errorbar(g_vSi[i],g_ratio[i],xerr = Ug_vSi[i],yerr = Ug_ratio[i], c = 'gray', capsize = 3, linestyle = '-', marker = 'o')
for i in range(np.size(b_vSi)):
	plt.errorbar(b_vSi[i],b_ratio[i],xerr = Ub_vSi[i],yerr = Ub_ratio[i], c = 'b', capsize = 3, linestyle = '-', marker = 'o')
for i in range(np.size(r_vSi)):
	plt.errorbar(r_vSi[i],r_ratio[i],xerr = Ur_vSi[i],yerr = Ur_ratio[i], c = 'r', capsize = 3, linestyle = '-', marker = 'o')
plt.plot(np.linspace(9,13,10), f_line(np.linspace(9,13,10), params_b[0], params_b[1]), c = 'b', linestyle = '--')
plt.plot(np.linspace(9,17,10), f_line(np.linspace(9,17,10), params_r[0], params_r[1]), c = 'r', linestyle = '--')
collection = collections.BrokenBarHCollection.span_where(np.linspace(9,17,100),ymin=0.016,ymax=0.06,where=np.ones(100)>0,facecolor='gray',alpha=0.5)
ax.add_collection(collection)
collection = collections.BrokenBarHCollection.span_where(np.linspace(9,17,100),ymin=0.064,ymax=0.104,where=np.ones(100)>0,facecolor='yellow',alpha=0.5)
ax.add_collection(collection)
ax.text(14,0.02,'sub-M$_{ch}$ Double Det.')
ax.text(15,0.09,'M$_{ch}$ Del. Det.')
plt.show()



plt.xlabel('The velocity of Si II at maximum [$\\rm 10^3\\ km\\ s^{-1}$]')
plt.ylabel('$Nebular velocity [$\\rm 10^3\\ km\\ s^{-1}$]')
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
plt.xlabel('Nebular velocity [$\\rm 10^3\\ km\\ s^{-1}$]')
plt.ylabel('$\\rm {\\Delta}m_{15}(B)$ [magnitude]')
for i in range(np.size(g_vSi)):
	plt.errorbar(g_vN[i],g_delta[i],xerr = Ug_vN[i],yerr = Ug_delta[i], c = 'gray', capsize = 3, linestyle = '-', marker = 'o')
for i in range(np.size(b_vSi)):
	plt.errorbar(b_vN[i],b_delta[i],xerr = Ub_vN[i],yerr = Ub_delta[i], c = 'b', capsize = 3, linestyle = '-', marker = 'o')
for i in range(np.size(r_vSi)):
	plt.errorbar(r_vN[i],r_delta[i],xerr = Ur_vN[i],yerr = Ur_delta[i], c = 'r', capsize = 3, linestyle = '-', marker = 'o')
plt.show()

fig, ax = plt.subplots()
plt.xlabel('$\\rm {\\Delta}m_{15}(B)$ [magnitude]')
plt.ylabel('$\\rm M_{Ni}/M_{Fe}$')
for i in range(np.size(g_vSi)):
	plt.errorbar(g_delta[i],g_ratio[i],xerr = Ug_delta[i],yerr = Ug_ratio[i], c = 'gray', capsize = 3, linestyle = '-', marker = sub_shape[g_subc[i]])
for i in range(np.size(b_delta)):
	plt.errorbar(b_delta[i],b_ratio[i],xerr = Ub_delta[i],yerr = Ub_ratio[i], c = 'b', capsize = 3, linestyle = '-', marker = sub_shape[b_subc[i]])
for i in range(np.size(r_vSi)):
	plt.errorbar(r_delta[i],r_ratio[i],xerr = Ur_delta[i],yerr = Ur_ratio[i], c = 'r', capsize = 3, linestyle = '-', marker = sub_shape[r_subc[i]])
collection = collections.BrokenBarHCollection.span_where(np.linspace(0.7,2,100),ymin=0.016,ymax=0.06,where=np.ones(100)>0,facecolor='gray',alpha=0.5)
ax.add_collection(collection)
collection = collections.BrokenBarHCollection.span_where(np.linspace(0.7,2,100),ymin=0.064,ymax=0.104,where=np.ones(100)>0,facecolor='yellow',alpha=0.5)
ax.add_collection(collection)
ax.text(1.6,0.02,'sub-M$_{ch}$ Double Det.')
ax.text(1.6,0.07,'M$_{ch}$ Del. Det.')
ax.text(1.75,0.04,'86G', c='b')
ax.text(1.96,0.0325,'03gs',c='r')
plt.show()

for i in range(np.size(jlist)):
	plt.scatter(delta1[jlist[i]], vSi1[jlist[i]])
plt.show()

ratio_M = []

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

'''
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