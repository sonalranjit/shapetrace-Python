import numpy as np


filename = 'export/depths1.txt'

data = np.loadtxt(filename)

print len(data)