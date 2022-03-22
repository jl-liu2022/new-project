import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

cm = plt.cm.get_cmap('Blues')

def rdata0(filename, start):
	with open(filename,'r') as f:
		for i in range (start):
			line = f.readline()
		nametemp = []
		ptemp = []
		dtemp = []
		Udtemp = []
		rtemp = []
		Mtemp = []
		vdottemp = []
		Uvdottemp = []
		vSitemp = []
		UvSitemp = []
		vFetemp = []
		vNitemp = []
		wFetemp = []
		wNitemp   = []
		ratio_flux = []
		ratio_high = []
		a = line.split()
		while line:
			a = line.split()
			nametemp.append(a[1])
			ptemp.append(float(a[2]))
			dtemp.append(float(a[5]))
			Udtemp.append(float(a[6]))
			rtemp.append(float(a[15]))
			Mtemp.append(float(a[7]))
			vdottemp.append(float(a[8]))
			Uvdottemp.append(float(a[9]))
			vSitemp.append(float(a[10]))
			UvSitemp.append(float(a[11]))
			vFetemp.append(float(a[16]))
			vNitemp.append(float(a[17]))
			wFetemp.append(float(a[18]))
			wNitemp.append(float(a[19]))
			ratio_flux.append(float(a[20]))
			ratio_high.append(float(a[21]))
			line = f.readline()
	number = len(ptemp)
	return nametemp, ptemp, dtemp, Udtemp, rtemp, Mtemp, vdottemp, Uvdottemp, vSitemp, UvSitemp, vFetemp, vNitemp, wFetemp, wNitemp, ratio_flux, ratio_high, number

def rU0(filename, start):
	with open(filename,'r') as f:
		for i in range (start):
			line = f.readline()
		UvFetemp = []
		UvNitemp = []
		UwFetemp = []
		UwNitemp   = []
		Uratio_flux = []
		Uratio_high = []
		a = line.split()
		while line:
			a = line.split()
			UvFetemp.append(float(a[4]))
			UvNitemp.append(float(a[6]))
			UwFetemp.append(float(a[3]))
			UwNitemp.append(float(a[5]))
			Uratio_flux.append(float(a[7]))
			Uratio_high.append(float(a[8]))
			line = f.readline()
	return UvFetemp, UvNitemp, UwFetemp, UwNitemp, Uratio_flux, Uratio_high


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
		a = line.split()
		while line:
			a = line.split()
			nametemp.append(a[0])
			ptemp.append(float(a[1]))
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
			line = f.readline()
	number = len(ptemp)
	return nametemp, ptemp, dtemp, Udtemp, vSitemp, UvSitemp, rtemp, vFetemp, vNitemp, wFetemp, wNitemp, ratio_flux, number

def rU(filename, start):
	with open(filename,'r') as f:
		for i in range (start):
			line = f.readline()
		UvFetemp = []
		UvNitemp = []
		UwFetemp = []
		UwNitemp   = []
		Uratio_flux = []
		a = line.split()
		while line:
			a = line.split()
			UvFetemp.append(float(a[2]))
			UvNitemp.append(float(a[3]))
			UwFetemp.append(float(a[4]))
			UwNitemp.append(float(a[5]))
			Uratio_flux.append(float(a[6]))
			line = f.readline()
	return UvFetemp, UvNitemp, UwFetemp, UwNitemp, Uratio_flux

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

plt.xlabel('The decline-rate parameter [magnitude]')
plt.ylabel('$M_{Ni}/M_{Fe}$')
for i in range(N):
	if delta[i] != 0.00:
		plt.errorbar(delta[i], ratio[i], xerr = Udelta[i],yerr = Uratio[i], capsize = 3, linestyle = '-', marker = 'o', c = 'b')
plt.scatter(0.6, 0.05, alpha = 0)
plt.legend(loc = 'upper left', frameon=False)
plt.show()
plt.xlabel('The decline-rate parameter [magnitude]')
plt.ylabel('$M_{Ni}/M_{Fe}$')
for i in range(N):
	if delta[i] != 0.00:
		plt.scatter(delta[i], ratio[i], c=phase[i], s = 20, vmin = 150, vmax = 430, cmap = cm)
plt.colorbar(label = 'phase since maximum')
plt.show()

plt.xlabel('The velocity of Si II at maximum [km/s]')
plt.ylabel('$M_{Ni}/M_{Fe}$')
for i in range(N):
	if v_SiII[i] != 0:
		plt.errorbar(v_SiII[i],ratio[i],xerr = Uv_SiII[i],yerr = Uratio[i], capsize = 3, marker = 'o', c = 'b')
plt.show()
plt.xlabel('The velocity of Si II at maximum [km/s]')
plt.ylabel('$M_{Ni}/M_{Fe}$')
for i in range(N):
	if v_SiII[i] != 0:
		plt.scatter(v_SiII[i], ratio[i], c=phase[i], s = 20, vmin = 150, vmax = 430, cmap = cm)
plt.colorbar(label = 'phase since maximum')
plt.show()

name1, phase1, delta1, Udelta1, vSi1, UvSi1, ratio1, vFe1, vNi1, wFe1, wNi1, r_flux1, number1 = rdata('result_data0_IMG.dat',1)
name1.append('tail')
UvFe1, UvNi1, UwFe1, UwNi1, U_r_flux1 = rU('Uncertenty_IMG.dat',1)

name2, phase2, delta2, Udelta2, ratio2, M_B2, vdot2, Uvdot2, vSi2, UvSi2, vFe2, vNi2, wFe2, wNi2, r_flux2, r_high2, number2 = rdata0('result_data.txt',3)
name2.append('tail')
UvFe2, UvNi2, UwFe2, UwNi2, U_r_flux2, U_r_high2 = rU0('Uncertenty.txt',3)

print(number1)

for i in range(number1-3):
	name2[i] = 'SN'+name2[i]
	ratio1[i] = ratio1[i]*58/56
	ratio2[i] = ratio2[i]*58/56

j0 = 0
for i in range(N):
	for j in range(j0, number1, 1):
		if name[i] == name1[j]:
			plt.errorbar(ratio1[j], ratio[i], xerr = 0.5*ratio1[j], yerr = Uratio[i], capsize = 3, linestyle = '-', marker = 'o', c = 'b')
			j0 = j+1
			break
plt.xlabel('Improved Multi-Gaussian')
plt.ylabel('Synthetic')
plt.plot(np.linspace(0,np.max(ratio),10), np.linspace(0,np.max(ratio),10), c = 'grey', linestyle = '--')
plt.scatter(0.05, 0.05, label = '$M_{Ni}/M_{Fe}$', alpha = 0)
plt.legend(loc = 'upper left', frameon=False)
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

g_vSi = []
Ug_vSi = []
g_ratio = []
g_delta = []
Ug_delta = []
b_vSi = []
Ub_vSi = []
b_ratio = []
b_delta = []
Ub_delta = []
r_vSi = []
Ur_vSi = []
r_ratio = []
r_delta = []
Ur_delta = []
for i in range(np.size(jlist)):
	if vFe1[jlist[i]]*vNi1[jlist[i]]<0:
		g_vSi.append(vSi1[jlist[i]])
		Ug_vSi.append(UvSi1[jlist[i]])
		g_ratio.append(ratio1[jlist[i]])
		g_delta.append(delta1[jlist[i]])
		Ug_delta.append(Udelta1[jlist[i]])
	elif vFe1[jlist[i]] < 0:
		b_vSi.append(vSi1[jlist[i]])
		Ub_vSi.append(UvSi1[jlist[i]])
		b_ratio.append(ratio1[jlist[i]])
		b_delta.append(delta1[jlist[i]])
		Ub_delta.append(Udelta1[jlist[i]])
	else:
		r_vSi.append(vSi1[jlist[i]])
		Ur_vSi.append(UvSi1[jlist[i]])
		r_ratio.append(ratio1[jlist[i]])
		r_delta.append(delta1[jlist[i]])
		Ur_delta.append(Udelta1[jlist[i]])
plt.xlabel('The velocity of Si II at maximum [$10^3$km/s]')
plt.ylabel('$M_{Ni}/M_{Fe}$')
for i in range(np.size(g_vSi)):
	plt.errorbar(g_vSi[i],g_ratio[i],xerr = Ug_vSi[i],yerr = g_ratio[i]*0.5, c = 'gray', capsize = 3, linestyle = '-', marker = 'o')
for i in range(np.size(b_vSi)):
	plt.errorbar(b_vSi[i],b_ratio[i],xerr = Ub_vSi[i],yerr = b_ratio[i]*0.5, c = 'b', capsize = 3, linestyle = '-', marker = 'o')
for i in range(np.size(r_vSi)):
	plt.errorbar(r_vSi[i],r_ratio[i],xerr = Ur_vSi[i],yerr = r_ratio[i]*0.5, c = 'r', capsize = 3, linestyle = '-', marker = 'o')
plt.show()

plt.xlabel('The decline-rate parameter [magnitude]')
plt.ylabel('$M_{Ni}/M_{Fe}$')
for i in range(np.size(g_vSi)):
	plt.errorbar(g_delta[i],g_ratio[i],xerr = Ug_delta[i],yerr = g_ratio[i]*0.5, c = 'gray', capsize = 3, linestyle = '-', marker = 'o')
for i in range(np.size(b_delta)):
	plt.errorbar(b_delta[i],b_ratio[i],xerr = Ub_delta[i],yerr = b_ratio[i]*0.5, c = 'b', capsize = 3, linestyle = '-', marker = 'o')
for i in range(np.size(r_vSi)):
	plt.errorbar(r_delta[i],r_ratio[i],xerr = Ur_delta[i],yerr = r_ratio[i]*0.5, c = 'r', capsize = 3, linestyle = '-', marker = 'o')
plt.show()

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

