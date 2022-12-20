import numpy as np
import pandas as pd
 
nameDict = ['SN', 'AS']
name = []
phase = []
ratio = []
Uratio_up = []
Uratio_low = []
with open('flor.dat','r') as f:
	line = f.readline()
	while(1):
		a = line.split()
		if not a:
			line = f.readline()
			continue
		#print(a[0])
		if a[0] == 'end':
			break
		if a[0][:2] in nameDict:
			nameTempt = a[0]
			name.append(a[0])
			phase.append(a[7][1:])
			ratioTempt = a[-1].split('+')
			ratio.append(ratioTempt[0])
			ratioTempt = ratioTempt[1].split('−')
			Uratio_up.append(ratioTempt[0])
			Uratio_low.append(ratioTempt[1])
		elif a[0][:1] == '+':
			name.append(nameTempt)
			phase.append(a[0][1:])
			ratioTempt = a[-1].split('+')
			ratio.append(ratioTempt[0])
			ratioTempt = ratioTempt[1].split('−')
			Uratio_up.append(ratioTempt[0])
			Uratio_low.append(ratioTempt[1])
		line = f.readline()

data = {'name':name,
		'phase':phase,
		'ratio':ratio,
		'Uratio_up':Uratio_up,
		'Uratio_low':Uratio_low}
df = pd.DataFrame(data)
df.to_csv('florSort', index=None)
