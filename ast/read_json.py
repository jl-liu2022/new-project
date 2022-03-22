import json
import numpy as np
import matplotlib.pyplot as plt

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
	redshift = json_data[filename]['spectra'][0]['redshift']
	host_name = json_data[filename]['host'][0]['value']
	host_name = remove_space(host_name)

#Output data
N = len(json_data[filename]['spectra'])
for i in range(N):
	MJD = json_data[filename]['spectra'][i]['time']
	phase = int(float(MJD) - float(maxMJD))
	if phase > 200 and phase < 430:
		print(i)
		data = json_data[filename]['spectra'][i]['data']
		instrument = json_data[filename]['spectra'][i]['instrument']
		instrument = remove_space(instrument)
		length = len(instrument)
		telescope = 'none'
		for j in range(length):
			if instrument[j] == '-':
				telescope = instrument[:j]
				instrument = instrument[(j+1):]
				break
		if telescope == 'none':
			telescope = input('telescope: ')
			instrument = input('instrument: ')
		filename_t = '/Users/pro/python/spectra_data/OSC/' + filename + '/' + filename + '_' + str(phase) + '.dat'
		xlist = []
		ylist = []
		with open(filename_t) as f:
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
		plt.title('%s at phase %s d' %(filename, phase))
		plt.xlabel('Wavelength [Å]')
		plt.ylabel('Normalized flux')
		plt.yticks([])
		plt.plot(xlist,ylist)
		plt.show()

	
	
