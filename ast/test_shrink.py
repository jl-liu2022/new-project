import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections

def Fe56(t, Ni, Co, lambda_Ni, lambda_Co):
	return Co + Ni - Co*np.exp(-lambda_Co*t) - lambda_Co*Ni/(lambda_Co-lambda_Ni)*np.exp(-lambda_Ni*t) + lambda_Ni*Ni/(lambda_Co-lambda_Ni)*np.exp(-lambda_Co*t)

ModelNameList = ['n1','n10', 'n100', 'n100h', 'n100l', 'n100_z0.01', 'n150', 'n1600', 'n1600c', 'n20', 'n200', 'n3', 'n300c', 'n40', 'n5']
Ni_58_List = []
Ni_56_List = []
Co_56_list = []
for item in ModelNameList:
	with open('/Users/pro/python/ast/models/ddt_2013_' + item + '_abundances.dat','r') as f:
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

lambda_Ni = 1/8.77
lambda_Co = 1/111


day_list = np.linspace(150,500,351)
fig, ax = plt.subplots()
for i in range(np.size(ModelNameList)):
	if ModelNameList[i] == 'n3':
		Ni_56 = Ni_56_List[i]
		Co_56 = Co_56_list[i]
		Ni_58 = Ni_58_List[i]
		ratio_n3_list = Ni_58 / Fe56(day_list, Ni_56, Co_56, lambda_Ni, lambda_Co)
	if ModelNameList[i] == 'n20':
		Ni_56 = Ni_56_List[i]
		Co_56 = Co_56_list[i]
		Ni_58 = Ni_58_List[i]
		ratio_n20_list = Ni_58 / Fe56(day_list, Ni_56, Co_56, lambda_Ni, lambda_Co)

C = 0.0585*Fe56(180,1,0,lambda_Ni,lambda_Co)
double_dt_list = C/Fe56(day_list,1,0,lambda_Ni,lambda_Co)

ax.fill_between(day_list, ratio_n3_list, ratio_n20_list, alpha=0.5, color = 'yellow')
ax.fill_between(day_list, np.zeros_like(day_list), double_dt_list, alpha=0.5, color = 'gray')
plt.show() 
