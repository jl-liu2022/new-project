import numpy as np
import matplotlib.pyplot as plt

cm = plt.cm.get_cmap('Blues')

def rdata(filename, start):
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

def rU(filename, start):
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

name = ['SN1990N', 'SN1991T','SN1993Z','SN1994ae','SN1995D','SN1996X','SN1998aq','SN1998bu','SN1999aa','SN2002bo','SN2002dj','SN2002er',
'SN2003cg','SN2003du','SN2003gs','SN2003hv','SN2003kf','SN2004eo','SN2005cf','SN2006X','SN2007af','SN2007le','SN2008Q',
'SN2009ig','SN2009le','SN2010ev','SN2010gp','SN2011at','SN2011by','SN2011ek','SN2011fe','SN2011im','SN2011iv',
'SN2012cg','SN2012fr','SN2012hr','SN2012ht','SN2013aa','SN2013cs','SN2013dy','SN2013gy','SN2014J','ASASSN14jg',
'SN2015F']
v_SiII = [10.53, 9.8, 0, 11.1, 10.1, 11.3, 10.7, 10.8, 10.5, 13.2, 13.4, 11.7,
10.9,10.4,11.4,11.3,11.1,10.7,10.1,16.1,10.8,12.9,11.09,
13.0,0,14.98,0,0,10.35,0,10.4,0,10.4,
10.0,11.93,11.5,11.0,10.2,12.5,10.3,10.7,12.1,0,
10.1]

Uv_SiII = [0.15, 0.2, 0, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.2, 0.2, 0.3,
0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.6, 0.10,
0.3, 0,0.02,0,0,0.14, 0,0.2, 0,0.2,
0.2, 0.1, 0.2, 0.2,0.2, 0.2, 0.2, 0.2, 0.2, 0,
0.2]

N = len(name)

delta = [1.09,0.97,0.87,1.04,0.90,1.20,1.11,1.05,0.90,1.10,1.02,1.23,
1.14,1.02,1.59,1.55,1.03,1.31,1.11,1.08,1.08,1.05,1.09,
0.88,0.91,1.12,1.10,0.92,1.11,1.00,1.18,1.06,1.63, 
0.98,0.90,1.07,1.56,0.90,0.81,0.94,1.10,1.01,0.89,
1.18]
Udelta = [0.06,0.06,0.06,0.06,0.06,0.06,0.07,0.06,0.06,0.06,0.08,0.06,
0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,
0.06,0.06,0.06,0.06,0.06,0.06,0.007,0.06,0.07,0.05,
0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,
0.02]

ratio = [0.023,0.031,0.033,0.050,0.008,0.053,0.037,0.059,0.055,0.051,0.051,0.083,
0.049,0.031,0.054,0.087,0.032,0.055,0.029,0.065,0.035,0.032,0.081,
0.028,0.036,0.044,0.033,0.059,0.039,0.020,0.041,0.047,0.051,
0.025,0.028,0.021,0.009,0.026,0.022,0.025,0.057,0.026,0.039,
0.050]
tUratio = [[0.005,0.011,0.007,0.009,0.006,0.020,0.011,0.008,0.012,0.010,0.010,0.019,
0.008,0.009,0.008,0.012,0.007,0.009,0.005,0.008,0.006,0.005,0.011,
0.005,0.006,0.007,0.005,0.010,0.006,0.005,0.006,0.013,0.018,
0.005,0.005,0.006,0.004,0.005,0.010,0.005,0.010,0.005,0.007,
0.006],
[0.004,0.009,0.006,0.008,0.006,0.018,0.010,0.008,0.010,0.009,0.008,0.016,
0.007,0.008,0.007,0.010,0.006,0.008,0.006,0.007,0.005,0.005,0.010,
0.005,0.005,0.006,0.005,0.009,0.006,0.006,0.005,0.011,0.015,
0.005,0.005,0.006,0.004,0.005,0.008,0.005,0.008,0.006,0.006,
0.006]]
Uratio = []
for i in range(N):
	Uratio.append([[tUratio[0][i]],[tUratio[1][i]]])
phase = [280,316,233,368,284.7,246,241.5,280.4,282.6,227.7,275,216,
385,272,201,323,397.3,228,319.6,277.6,301,304.7,201.1,
405,324,272,279,349,310,423,311,314,318,
279,290,283,433,344,300,333,280,282,323,
280]


plt.xlabel('The decline-rate parameter [magnitude]')
plt.ylabel('Number ratio of Ni to Fe')
for i in range(N):
	plt.errorbar(delta[i],ratio[i],xerr = Udelta[i],yerr = Uratio[i], capsize = 3, linestyle = '-', marker = 'o', ecolor = 'y', fmt = 'none', alpha = 0.5)
plt.scatter(delta, ratio, c=phase, s = 20, vmin = 150, vmax = 430, cmap = cm)
plt.colorbar(label = 'phase since maximum')
plt.show()

plt.xlabel('The velocity of Si II at maximum [km/s]')
plt.ylabel('Number ratio of Ni to Fe')
for i in range(N):
	if v_SiII[i] != 0:
		plt.errorbar(v_SiII[i],ratio[i],xerr = Uv_SiII[i],yerr = Uratio[i], capsize = 3, linestyle = '-', marker = 'o', ecolor = 'y', fmt = 'none', alpha = 0.5)
		plt.scatter(v_SiII[i], ratio[i], c=phase[i], s = 20, vmin = 150, vmax = 430, cmap = cm)
plt.colorbar(label = 'phase since maximum')
plt.show()

name1, phase1, delta1, Udelta1, ratio1, M_B1, vdot1, Uvdot1, vSi1, UvSi1, vFe1, vNi1, wFe1, wNi1, r_flux1, r_high1, number1 = rdata('result_data.txt',3)
name1.append('tail')
UvFe1, UvNi1, UwFe1, UwNi1, U_r_flux1, U_r_high1 = rU('Uncertenty.txt',3)

for i in range(number1):
	name1[i] = 'SN'+name1[i]

j0 = 0
for i in range(N):
	for j in range(j0, number1, 1):
		if name[i] == name1[j]:
			plt.errorbar(ratio1[j], ratio[i], xerr = 0.5*ratio1[j], yerr = Uratio[i], capsize = 3, linestyle = '-', marker = 'o', c = 'b')
			j0 = j+1
			break
plt.xlabel('Multi-Gaussian ratio x')
plt.ylabel('Synthetic ratio y')
plt.plot(np.linspace(0,np.max(ratio),10), np.linspace(0,np.max(ratio),10), label = 'y = x', c = 'r')
plt.legend(loc = 'upper left')
plt.show()
			


