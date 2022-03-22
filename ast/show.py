import numpy as np
import matplotlib.pyplot as plt

while(1):
	filename = input('input the name of the data file: ')
	start =int(input('input the number of line you want to start: '))
	with open('/Users/pro/python/spectra_data/' + filename,'r') as f:
		for i in range (start):
			line = f.readline()
		xlist = []
		ylist = []
		while line:
			a = line.split()
			xlist.append(float(a[0]))
			ylist.append(float(a[1]))
			line = f.readline()
	xlist = np.array(xlist)
	ylist = np.array(ylist)

	plt.xlabel('Wavelength [Å]')
	plt.ylabel('Normalized flux')
	plt.yticks([])
	plt.plot(xlist,ylist)
	plt.show()
