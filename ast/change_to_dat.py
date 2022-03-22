import numpy as np
import matplotlib.pyplot as plt

def cut(string):
	n = 0
	pos = [-1]
	for i in range(len(string)):
		if string[i] == ',':
			n += 1
			pos.append(i) 
	cut_string = []

	for i in range(n):
		cut_string.append(string[(pos[i]+1):(pos[i+1])])
	cut_string.append(string[(pos[n]+1):(len(string)-1)])
	return cut_string

def rdata(filename, start):
	with open(filename,'r') as f:
		for i in range (start):
			line = f.readline()
		Wavelength = []
		Flux = []
		SkyFlux = []
		a = cut(line)
		print(a)
		while line:
			a = cut(line)
			Wavelength.append(float(a[0]))
			Flux.append(float(a[1]))
			SkyFlux.append(float(a[3]))
			line = f.readline()
	number = len(Wavelength)
	return Wavelength, Flux, SkyFlux, number

filename = input('input the filename: ')
Wavelength, Flux, SkyFlux, number = rdata(filename, 2)

with open(filename+'.dat', 'w') as f1:
	for i in range(number):
		f1.writelines('%s %s %s\n' %(Wavelength[i], Flux[i], SkyFlux[i]))



