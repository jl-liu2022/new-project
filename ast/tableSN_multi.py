import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
	return nametemp, ptemp, dtemp, Udtemp, redshift, ebv, vSitemp, UvSitemp, rtemp, vFetemp, vNitemp, wFetemp, wNitemp, ratio_flux, number

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

integrate = 0.5372471812372228
mean = 0.5023869679593065
save_as = input('method: ')
name, phase, delta, Udelta, redshift, ebv, vSi, UvSi, ratio, vFe, vNi, wFe, wNi, r_flux, number = rdata('result_data0_'+save_as+'.dat',1)
name.append('tail')

UvFe, UvNi, UwFe, UwNi, U_r_flux = rU('Uncertenty_'+save_as+'.dat',1)
ref_spectra = [1,2,2,2,'BSNIP','BSNIP','BSNIP',3,6,'BSNIP',7,'CfA',8,'BSNIP','BSNIP','BSNIP','CfA','BSNIP','BSNIP',9,9,10,10,10,10,10,
11,12,13,13,13,13,13,14,14,12,15,13,10,10,17,14,19,20,19,21,22,22,'This work',19,14,23]
U_ratio = []
for i in range(number):
	ratio[i] *= 58/56*mean/integrate
	u1 = 0.4
	u2 = U_r_flux[i]/r_flux[i]
	UvFe[i] += 200
	UvNi[i] += 200
	U_ratio_t = np.sqrt(u1**2 + u2**2)*ratio[i]
	U_ratio.append(U_ratio_t)

Neb_v = (np.array(vFe)+np.array(vNi))/2
U_Neb_v = np.sqrt(np.array(UvFe)**2+np.array(UvNi)**2)/2
with open('table'+save_as+'.dat','w') as fw:
	for i in range(number):
		fw.writelines('%s & +%d & %d$\\pm$%d & %d$\\pm$%d & %d$\\pm$%d & %d$\\pm$%d & %.3f$\\pm$%.3f & %d$\\pm$%d & %.3f$\\pm$%.3f & %s \\\\ \n' %(name[i],phase[i],vFe[i],UvFe[i],vNi[i],UvNi[i],wFe[i],UwFe[i],wNi[i],UwNi[i],r_flux[i],U_r_flux[i],Neb_v[i],U_Neb_v[i],ratio[i],U_ratio[i],ref_spectra[i]))

ref_Si = []
ref_LC = []
				
'''
data = {'SN Name':[name[i] for i in range(number)],
		'Phase[days]':['%d'%phase[i] for i in range(number)],
		'[Fe II] Velocity[km/s]':['%d±%d'%(vFe[i],UvFe[i]) for i in range(number)],
		'[Ni II] Velocity[km/s]':['%d±%d'%(vNi[i],UvNi[i]) for i in range(number)],
		'[Fe II] FWHM[km/s]':['%d±%d'%(wFe[i],UwFe[i]) for i in range(number)],
		'[Ni II] FWHM[km/s]':['%d±%d'%(wNi[i],UwNi[i]) for i in range(number)],
		'Integrated Flux Ratio Ni/Fe':['%.3f±%.3f'%(r_flux[i],U_r_flux[i]) for i in range(number)],
		'Height Ratio Ni/Fe':['%.3f±%.3f'%(r_high[i],U_r_high[i]) for i in range(number)]}
df = pd.DataFrame(data)

writer = pd.ExcelWriter('my.xlsx')
df.to_excel(writer)
writer.save()
'''