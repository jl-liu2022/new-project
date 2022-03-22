import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def remove_space(string):
	N = len(string)
	new_string = ''
	for i in range(N):
		if string[i] != ' ':
			new_string = new_string + string[i]
	return new_string

#Read file
filename = input('filename: ')
with open('/Users/pro/python/spectra_data/OSC/' + filename + '/' + filename + '.json','r') as f:
	json_data = json.load(f)
	maxMJD = input('maxMJD: ')
	ebv = json_data[filename]['ebv'][0]['value']
	ebv_error = json_data[filename]['ebv'][0]['e_value']
	redshift = float(input('redshift: '))
	host_name = json_data[filename]['host'][0]['value']
	host_name = remove_space(host_name)

#Output data
N = len(json_data[filename]['spectra'])
print(N)
for i in range(N):
	MJD = json_data[filename]['spectra'][i]['time']
	phase = int(float(MJD) - float(maxMJD))
	if phase > 200 and phase < 430:
		print(i)
		print('phase: %s' %phase)
		instrument = 'none'
		telescope = 'none'
		data = json_data[filename]['spectra'][i]['data']
		if instrument != 'none':
			instrument = json_data[filename]['spectra'][i]['instrument']
			instrument = remove_space(instrument)
			length = len(instrument)
			for j in range(length):
				if instrument[j] == '-':
					telescope = instrument[:j]
					instrument = instrument[(j+1):]
					break
			if telescope == 'none':
				telescope = input('telescope: ')
				instrument = input('instrument: ')
		else:
			telescope = input('telescope: ')
			instrument = input('instrument: ')
		filename_t = '/Users/pro/python/spectra_data/OSC/' + filename + '/' + filename + '_' + str(phase) + '.dat'
		xlist = []
		ylist = []
		with open(filename_t,'w') as f:
			f.writelines('OSC indice %d\n' %i)
			f.writelines('maxMJD %s\n' %maxMJD)
			f.writelines('MJD %s\n' %MJD)
			f.writelines('ebv %s\n' %ebv)
			f.writelines('ebv_error %s\n' %ebv_error)
			f.writelines('redshift %s\n' %redshift)
			f.writelines('host_name %s\n' %host_name)
			f.writelines('telescope %s\n' %telescope)
			f.writelines('instrument %s\n' %instrument)
			column = len(data[0])
			if column == 3:
				for k in range(len(data)):
					f.writelines('%s %s %s\n' %(data[k][0], data[k][1], data[k][2]))
					xlist.append(float(data[k][0]))
					ylist.append(float(data[k][1]))
			elif column == 2:
				for k in range(len(data)):
					f.writelines('%s %s\n' %(data[k][0], data[k][1]))
					xlist.append(float(data[k][0]))
					ylist.append(float(data[k][1]))
			else:
				print('strange')
		resolution = int(xlist[1]) - int(xlist[0])
		print('resolution: %d' %resolution)
		plt.title('%s at phase %s d' %(filename, phase))
		plt.xlabel('Wavelength [Å]')
		plt.ylabel('Normalized flux')
		plt.yticks([])
		plt.plot(xlist,ylist, c = 'b')
		plt.show()
		while(1):
			width = int(input('width: '))
			s_ylist = signal.savgol_filter(ylist,width,1)
			plt.title('%s at phase %s d' %(filename, phase))
			plt.xlabel('Wavelength [Å]')
			plt.ylabel('Normalized flux')
			plt.yticks([])
			plt.plot(xlist,s_ylist, c = 'b')
			plt.show()
			go = input('ok?(y/n): ')
			if go == 'y':
				break
		


	
	
