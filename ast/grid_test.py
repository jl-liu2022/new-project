import numpy as np

def Append(l1, l2):
	l3 = []
	for item in l1:
		l3.append(item)
	for item in l2:
		l3.append(item)
	return np.array(l3)

def grid2(list1, list2):
	shape1 = np.shape(list1)
	shape2 = np.shape(list2)
	if np.size(shape1) == 1:
		for i in range(shape1[0]):
			list1[i] = [list1[i]]
	if np.size(shape2) == 1:
		for i in range(shape2[0]):
			list2[i] = [list2[i]]
	list3 = []
	for i in range(shape1[0]):
		for j in range(shape2[0]):
			list3.append(Append(list1[i],list2[j]))
	return np.array(list3)

def make_grid(lists):
	no_of_list = np.shape(lists)[0]
	list_temp = lists[0]
	for i in range(1, no_of_list):
		list_temp = grid2(list_temp, lists[i])
	return list_temp

size_v = 2
size_w = 3
bounds = [(-10000,10000),(-10000,10000),(1,20000),(1,20000),(0,np.inf),(0,np.inf)]
list_v1 = [-8000+1000*i for i in range(size_v)]
list_v2 = [-8000+1000*i for i in range(size_v)]
list_w1 = [500+1000*i for i in range(size_w)]
list_w2 = [500+1000*i for i in range(size_w)]
lists = [list_v1, list_v2, list_w1, list_w2]
para_grid = make_grid(lists)
append_list = np.ones([size_v*size_v*size_w*size_w,2])
para_grid = np.append(para_grid, append_list, axis=1)
print(para_grid)