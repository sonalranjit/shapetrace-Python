"""
Simple demo of a scatter plot.
"""
import numpy as np
import matplotlib.pyplot as plt


N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radiuses
label = ['1','2','3']
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
for i, txt in enumerate(label):
    plt.annotate(txt, (x[i],y[i]))
plt.show()