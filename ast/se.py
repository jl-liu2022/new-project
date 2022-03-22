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

def pclassify(p):
	if p < 250:
		c = 'purple'
	elif p < 300:
		c = 'g'
	elif p < 400:
		c = 'r'
	else:
		c = 'black'
	return c

name, phase, delta, Udelta, ratio, M_B, vdot, Uvdot, vSi, UvSi, vFe, vNi, wFe, wNi, r_flux, r_high, number = rdata('result_data.txt',3)
for i in range(49):
	name[i] = 'SN '+name[i]
	ratio[i] *= 58/56;
name.append('tail')
UvFe, UvNi, UwFe, UwNi, U_r_flux, U_r_high = rU('Uncertenty.txt',3)
print(number)

marker = ['.','o','v','^','<','>','s','p','x','d','_','*','h','H','+','x','D','d','_','.','o','v','^','<','>','s','p','*','h','H','+','x','D','x','d','_']
'''
plt.title('without wNi > 8400')
plt.xlabel('delta15')
plt.ylabel('ratio')
plt.scatter(0, 0, label = '<250', c = 'purple', s = 5)
plt.scatter(0, 0, label = '<300', c = 'g', s = 5)
plt.scatter(0, 0, label = '<400', c = 'r', s = 5)
plt.scatter(0, 0, label = '>400', c = 'black', s = 5)
for i in range(number):
	if wNi[i] < 7915:
		plt.scatter(delta[i], ratio[i], c = pclassify(phase[i]), s = 5)
plt.legend(loc = 'upper left')
plt.show()

plt.title('with wNi > 7915')
plt.xlabel('delta15')
plt.ylabel('ratio')
plt.scatter(0, 0, label = '<250', c = 'purple', s = 5)
plt.scatter(0, 0, label = '<300', c = 'g', s = 5)
plt.scatter(0, 0, label = '<400', c = 'r', s = 5)
plt.scatter(0, 0, label = '>400', c = 'black', s = 5)
for i in range(number):
	plt.scatter(delta[i], ratio[i], c = pclassify(phase[i]), s = 5)
plt.legend(loc = 'upper left')
plt.show()
'''
'''
name, phase, delta, ratio, vFe, vNi, wFe, wNi, number = rdata('result_data.txt',2)
UvFe, UvNi, UwFe, UwNi = rU('Uncertenty.txt',2)
'''
np.random.seed(399991)
plt.title('Velocity evolution of Fe[II]')
plt.xlabel('Phase [Days Since Peak Brightness]')
plt.ylabel('Line Velocity [km/s]')
plt.plot(np.linspace(150,450,100),np.zeros(100),linestyle='--',c = 'black')
head = 0
n = 0
for i in range(1,number+1):
	if name[head] != name[i]:
		plt.errorbar(phase[head:i],vFe[head:i],yerr = UvFe[head:i],label = 'v_Fe,%s'%name[head], capsize = 3, linestyle = '-', c = (np.random.rand(),np.random.rand(),np.random.rand()),marker = marker[n])
		head = i
		n += 1
plt.legend(loc = 'upper left')
plt.show()

np.random.seed(399991)



plt.title('Velocity evolution of Ni[II]')
plt.xlabel('Phase [Days Since Peak Brightness]')
plt.ylabel('Line Velocity [km/s]')
plt.plot(np.linspace(150,450,100),np.zeros(100),linestyle='--',c = 'black')
head = 0
n = 0
for i in range(1,number+1):
	if name[head] != name[i]:
		plt.errorbar(phase[head:i],vNi[head:i],yerr = UvNi[head:i],label = 'v_Ni,%s'%name[head], capsize = 3, linestyle = '-', c = (np.random.rand(),np.random.rand(),np.random.rand()),marker = marker[n])
		head = i
		n += 1
plt.legend(loc = 'upper left')
plt.show()

np.random.seed(399991)
plt.title('FWHM evolution of Fe[II]')
plt.xlabel('Phase [Days Since Peak Brightness]')
plt.ylabel('Line FWHM [km/s]')
plt.plot(np.linspace(150,450,100),np.zeros(100),linestyle='--',c = 'black')
head = 0
n = 0
for i in range(1,number+1):
	if name[head] != name[i]:
		plt.errorbar(phase[head:i],wFe[head:i],yerr = UwFe[head:i],label = 'FWHM_Fe,%s'%name[head], capsize = 3, linestyle = '-', c = (np.random.rand(),np.random.rand(),np.random.rand()),marker = marker[n])
		head = i
		n += 1
plt.legend(loc = 'upper left')
plt.show()

np.random.seed(399991)
plt.title('FWHM evolution of Ni[II]')
plt.xlabel('Phase [Days Since Peak Brightness]')
plt.ylabel('Line FWHM [km/s]')
plt.plot(np.linspace(150,450,100),np.zeros(100),linestyle='--',c = 'black')
head = 0
n = 0
for i in range(1,number+1):
	if name[head] != name[i]:
		plt.errorbar(phase[head:i],wNi[head:i],yerr = UwNi[head:i],label = 'FWHM_Ni,%s'%name[head], capsize = 3, linestyle = '-', c = (np.random.rand(),np.random.rand(),np.random.rand()),marker = marker[n])
		head = i
		n += 1
plt.legend(loc = 'upper left')
plt.show()

np.random.seed(399991)
plt.title('Number ratio of Ni to Fe evolution')
plt.xlabel('Phase [Days Since Peak Brightness]')
plt.ylabel('Ratio')
plt.plot(np.linspace(150,450,100),np.zeros(100),linestyle='--',c = 'black')
head = 0
n = 0
for i in range(1,number+1):
	if name[head] != name[i]:
		plt.errorbar(phase[head:i],ratio[head:i],yerr = np.array(ratio[head:i])*0.4,label = '%s'%name[head], capsize = 3, linestyle = '-', c = (np.random.rand(),np.random.rand(),np.random.rand()),marker = marker[n])
		head = i
		n += 1
plt.legend(loc = 'upper left')
plt.show()

np.random.seed(399991)
plt.title('Flux ratio [Ni II] 7155 to [Fe II] 7378 evolution')
plt.xlabel('Phase [Days Since Peak Brightness]')
plt.ylabel('Ratio')
plt.plot(np.linspace(150,450,100),np.zeros(100),linestyle='--',c = 'black')
head = 0
n = 0
for i in range(1,number+1):
	if name[head] != name[i]:
		plt.errorbar(phase[head:i],r_flux[head:i],yerr = np.array(U_r_flux[head:i]),label = '%s'%name[head], capsize = 3, linestyle = '-', c = (np.random.rand(),np.random.rand(),np.random.rand()),marker = marker[n])
		head = i
		n += 1
plt.legend(loc = 'upper left')
plt.show()

np.random.seed(399991)
plt.title('Height ratio [Ni II] 7155 to [Fe II] 7378 evolution')
plt.xlabel('Phase [Days Since Peak Brightness]')
plt.ylabel('Ratio')
plt.plot(np.linspace(150,450,100),np.zeros(100),linestyle='--',c = 'black')
head = 0
n = 0
for i in range(1,number+1):
	if name[head] != name[i]:
		plt.errorbar(phase[head:i],r_high[head:i],yerr = np.array(U_r_high[head:i]),label = '%s'%name[head], capsize = 3, linestyle = '-', c = (np.random.rand(),np.random.rand(),np.random.rand()),marker = marker[n])
		head = i
		n += 1
plt.legend(loc = 'upper left')
plt.show()

np.random.seed(399991)
plt.xlabel('The decline-rate parameter [magnitude]')
plt.ylabel('Number ratio of Ni to Fe')
head = 0
n = 0
for i in range(1,number+1):
	if name[head] != name[i]:
		if head + 1 != i:
			for j in range(head, i-1):
				d_to_300_first = np.abs(phase[j]-300)
				d_to_300_second = np.abs(phase[j+1]-300)
				if d_to_300_first >= d_to_300_second:
					if j + 2 == i:
						head = j + 1
						break
				else:
					head = j
					break
		plt.errorbar(delta[head],ratio[head],xerr = Udelta[head],yerr = ratio[head]*0.4, capsize = 3, linestyle = '-', ecolor = 'y', fmt = 'none', alpha = 0.5)
		plt.scatter(delta[head],ratio[head],c = phase[head], s = 20, vmin = 150, vmax = 450, cmap = cm)
		head = i
		n += 1
plt.colorbar(label = 'phase since maximum')
plt.show()

np.random.seed(399991)
plt.xlabel('The decline-rate parameter [magnitude]')
plt.ylabel('Flux ratio [Ni II] 7155 to [Fe II] 7378')
head = 0
n = 0
for i in range(1,number+1):
	if name[head] != name[i]:
		if head + 1 != i:
			for j in range(head, i-1):
				d_to_300_first = np.abs(phase[j]-300)
				d_to_300_second = np.abs(phase[j+1]-300)
				if d_to_300_first >= d_to_300_second:
					if j + 2 == i:
						head = j + 1
						break
				else:
					head = j
					break
		plt.errorbar(delta[head],r_flux[head],xerr = Udelta[head],yerr = U_r_flux[head], capsize = 3, linestyle = '-', ecolor = 'y', fmt = 'none', alpha = 0.5)
		plt.scatter(delta[head],r_flux[head],c = phase[head], s = 20, vmin = 150, vmax = 450, cmap = cm)
		head = i
		n += 1
plt.colorbar(label = 'phase since maximum')
plt.show()

np.random.seed(399991)
plt.xlabel('The decline-rate parameter [magnitude]')
plt.ylabel('Height ratio [Ni II] 7155 to [Fe II] 7378')
head = 0
n = 0
for i in range(1,number+1):
	if name[head] != name[i]:
		if head + 1 != i:
			for j in range(head, i-1):
				d_to_300_first = np.abs(phase[j]-300)
				d_to_300_second = np.abs(phase[j+1]-300)
				if d_to_300_first >= d_to_300_second:
					if j + 2 == i:
						head = j + 1
						break
				else:
					head = j
					break
		plt.errorbar(delta[head],r_high[head],xerr = Udelta[head],yerr = U_r_high[head], capsize = 3, linestyle = '-', ecolor = 'y', fmt = 'none', alpha = 0.5)
		plt.scatter(delta[head],r_high[head],c = phase[head], s = 20, vmin = 150, vmax = 450, cmap = cm)
		head = i
		n += 1
plt.colorbar(label = 'phase since maximum')
plt.show()

np.random.seed(399991)
plt.xlabel('The maximum magnitude in B-band? [magnitude]')
plt.ylabel('Number ratioof Ni to Fe')
head = 0
n = 0
for i in range(1,number+1):
	if name[head] != name[i]:
		if head + 1 != i:
			for j in range(head, i-1):
				d_to_300_first = np.abs(phase[j]-300)
				d_to_300_second = np.abs(phase[j+1]-300)
				if d_to_300_first >= d_to_300_second:
					if j + 2 == i:
						head = j + 1
						break
				else:
					head = j
					break
		plt.errorbar(M_B[head],ratio[head],yerr = ratio[head]*0.4,label = '%s at %d'%(name[head], phase[head]), capsize = 3, linestyle = '-', c = (np.random.rand(),np.random.rand(),np.random.rand()),marker = marker[n])
		head = i
		n += 1
plt.legend(loc = 'upper right')
plt.show()

np.random.seed(399991)
plt.xlabel('Si II Velocity at max. [1000km/s]')
plt.ylabel('Number ratioof Ni to Fe')
plt.plot(np.linspace(7,17,100),np.zeros(100),linestyle='--',c = 'black')
head = 0
n = 0
for i in range(1,number+1):
	if name[head] != name[i]:
		if head + 1 != i:
			for j in range(head, i-1):
				d_to_300_first = np.abs(phase[j]-300)
				d_to_300_second = np.abs(phase[j+1]-300)
				if d_to_300_first >= d_to_300_second:
					if j + 2 == i:
						head = j + 1
						break
				else:
					head = j
					break
		if vSi[head] != 0.0:
			plt.errorbar(vSi[head],ratio[head],xerr = UvSi[head],yerr = 0.5*ratio[head], capsize = 3, linestyle = '-', ecolor = 'y', fmt = 'none', alpha = 0.5)
			plt.scatter(vSi[head],ratio[head],c = phase[head], s = 20, vmin = 150, vmax = 450, cmap = cm)
		head = i
		n += 1
plt.colorbar(label = 'phase since maximum')
plt.show()

np.random.seed(399991)
plt.xlabel('Si II velocity gradient near max. [km/s]')
plt.ylabel('Number ratio of Ni to Fe')
plt.plot(np.linspace(0,300,100),np.zeros(100),linestyle='--',c = 'black')
head = 0
n = 0
for i in range(1,number+1):
	if name[head] != name[i]:
		if head + 1 != i:
			for j in range(head, i-1):
				d_to_300_first = np.abs(phase[j]-300)
				d_to_300_second = np.abs(phase[j+1]-300)
				if d_to_300_first >= d_to_300_second:
					if j + 2 == i:
						head = j + 1
						break
				else:
					head = j
					break
		if vdot[head] != 0.0:
			plt.errorbar(vdot[head],ratio[head],label = '%s at %d'%(name[head], phase[head]), capsize = 3, linestyle = '-', c = (np.random.rand(),np.random.rand(),np.random.rand()),marker = marker[n])
		head = i
		n += 1
plt.legend(loc = 'upper right')
plt.show()

np.random.seed(399991)
plt.xlabel('[Fe II] 7155 Å velocity [km/s]')
plt.ylabel('Si II velocity near max. [1000km/s]')
head = 0
n = 0
for i in range(1,number+1):
	if name[head] != name[i]:
		if head + 1 != i:
			for j in range(head, i-1):
				d_to_300_first = np.abs(phase[j]-300)
				d_to_300_second = np.abs(phase[j+1]-300)
				if d_to_300_first >= d_to_300_second:
					if j + 2 == i:
						head = j + 1
						break
				else:
					head = j
					break
		if vSi[head] != 0:
			plt.errorbar(vFe[head],vSi[head],xerr = UvFe[head], yerr = UvSi[head],label = '%s at %d'%(name[head], phase[head]), capsize = 3, linestyle = '-', c = (np.random.rand(),np.random.rand(),np.random.rand()),marker = marker[n])
		head = i
		n += 1
plt.legend(loc = 'upper left')
plt.show()

np.random.seed(399991)
plt.xlabel('Ni II] 7378 Å velocity [km/s]')
plt.ylabel('Si II velocity gradient near max. [1000 km/s]')
head = 0
n = 0
for i in range(1,number+1):
	if name[head] != name[i]:
		if head + 1 != i:
			for j in range(head, i-1):
				d_to_300_first = np.abs(phase[j]-300)
				d_to_300_second = np.abs(phase[j+1]-300)
				if d_to_300_first >= d_to_300_second:
					if j + 2 == i:
						head = j + 1
						break
				else:
					head = j
					break
		if vSi[head] != 0:
			plt.errorbar(vNi[head],vSi[head],xerr = UvNi[head], yerr = UvSi[head],label = '%s at %d'%(name[head], phase[head]), capsize = 3, linestyle = '-', c = (np.random.rand(),np.random.rand(),np.random.rand()),marker = marker[n])
		head = i
		n += 1
plt.legend(loc = 'upper left')
plt.show()
'''
while(1):
	n = []
	while(1):
		ntemp = int(input('to plot: '))
		n.append(ntemp)
		go = int(input('go?: '))
		if go == 0:
			break
	plt.title(name[n[0]])
	plt.xlabel('phase/d')
	plt.ylabel('v km/s')
	for i in range(len(n)):
		plt.scatter(phase[n[i]], vFe[n[i]], c = 'b')
		plt.scatter(phase[n[i]], vNi[n[i]], c = 'black')
	plt.legend(loc = 'upper left')
	plt.show()

while(1):
	n = []
	while(1):
		ntemp = int(input('to plot: '))
		n.append(ntemp)
		go = int(input('go?: '))
		if go == 0:
			break
	plt.title(name[n[0]])
	plt.xlabel('phase/d')
	plt.ylabel('relative value')
	plt.scatter(150, 0, label = 'Ni/Fe', c = 'r')
	plt.scatter(150, 0, label = 'vshift_Fe, 40000km/s', c = 'b')
	plt.scatter(150, 0, label = 'vshift_Ni, 40000km/s', c = 'black')
	plt.scatter(150, 0, label = 'FWHM_Fe, 40000km/s', c = 'purple')
	plt.scatter(150, 0, label = 'FWHM_Ni, 40000km/s', c = 'g')
	for i in range(len(n)):
		plt.scatter(phase[n[i]], ratio[n[i]], c = 'r')
		plt.scatter(phase[n[i]], vFe[n[i]]/40000, c = 'b')
		plt.scatter(phase[n[i]], vNi[n[i]]/40000, c = 'black')
		plt.scatter(phase[n[i]], wFe[n[i]]/40000, c = 'purple')
		plt.scatter(phase[n[i]], wNi[n[i]]/40000, c = 'g')
	plt.legend(loc = 'upper left')
	plt.show()
'''

 

