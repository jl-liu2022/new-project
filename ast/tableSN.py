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

save_as = input('method: ')
name, phase, delta, Udelta, ratio, M_B, vdot, Uvdot, vSi, UvSi, vFe, vNi, wFe, wNi, r_flux, r_high, number = rdata('result_data'+save_as+'.txt',3)
name.append('tail')
UvFe, UvNi, UwFe, UwNi, U_r_flux, U_r_high = rU('Uncertenty'+save_as+'.txt',3)
ref = [1,2,2,2,'BSNIP','BSNIP','BSNIP',3,4,5,6,6,'BSNIP',7,'CfA',8,'BSNIP','BSNIP','BSNIP','CfA','BSNIP','BSNIP',9,9,10,10,10,10,10,
11,12,13,13,13,13,13,13,12,14,14,14,12,15,13,16,13,10,17,10,17,18,14,19,20,19,21,22,22,19,14,23]
with open('/Users/pro/python/ast/table'+save_as,'w') as fw:
	for i in range(number-3):
		name[i] = 'SN'+name[i]
	for i in range(number):
		fw.writelines('%s & %d & %d$\\pm$%d & %d$\\pm$%d & %d$\\pm$%d & %d$\\pm$%d & %.3f$\\pm$%.3f & %s \\\\ \n' %(name[i],phase[i],vFe[i],UvFe[i],vNi[i],UvNi[i],wFe[i],UwFe[i],wNi[i],UwNi[i],r_flux[i],U_r_flux[i],ref[i]))


				
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