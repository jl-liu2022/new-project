import numpy as np

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
			WidthList.append(a[13])
			ESList.append(a[14])
			line = f.readline()
	return NameList, PhaseList, Min1List, Max1List, Min2List, Max2List,Min3List, Max3List, WidthList, ESList

list_name = 'Uncertenty_IMG.dat'
NameList, PhaseList, Min1List, Max1List, Min2List, Max2List,Min3List, Max3List, WidthList, ESList = read_list(list_name, 1)
list_size = np.size(NameList)

with open('FigureName.txt','w') as f:
	for i in range(list_size):
		if i%4 == 0:
			f.writelines('\\gridline{\\fig{'+NameList[i]+'_'+PhaseList[i]+'.pdf}{0.235\\textwidth}{}\n')
		else:
			f.writelines('           \\fig{'+NameList[i]+'_'+PhaseList[i]+'.pdf}{0.235\\textwidth}{}\n')
		if i%4 == 3:
			f.writelines('          }\n')

with open('FigureName_grad.txt','w') as f:
	for i in range(list_size):
		f.writelines('\\includegraphics[width=0.2\\linewidth]{'+NameList[i]+'_'+PhaseList[i]+'.pdf}\n')
		if i%4 == 3:
			f.writelines('\\\\\n')


