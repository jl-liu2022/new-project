import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

while(1):
	filename = input('filename: ')
	lena = mpimg.imread(filename)
	plt.imshow(lena)
	plt.axis('off')
	plt.show()

	while(1):
		degree = int(input('rotate: '))
		if degree == 0:
			break
		rotated_lena = ndimage.rotate(lena, degree)
		plt.imshow(rotated_lena)
		plt.axis('off')
		plt.show()