import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

k=0
while k in tqdm(range(1000)):
    if 0.5 < np.random.rand():
        k += 1
