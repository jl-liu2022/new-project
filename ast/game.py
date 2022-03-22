import numpy as np

def guess(change):
	a = 1
	b = np.random.randint(1,4)
	if a == b:
		if np.random.rand() < 0.5:
			c = 2
			if change == 1:
				a = 3
		else:
			c = 3
			if change == 1:
				a = 2
	elif b == 2:
		c = 3
		if change == 1:
			b = 1
	else:
		c = 2
		if change == 1:
			b = 1
	if a == b:
		result = 1
	else:
		result = 0
	return result

n_change = 0
n_unchange = 0
for i in range(10000):
	n_change += guess(1)
	n_unchange += guess(0)
print(n_change/10000)
print(n_unchange/10000)

