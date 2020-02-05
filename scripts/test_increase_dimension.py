import numpy as np
import matplotlib.pyplot as plt

from utils import increaseDimensionMatrix

data = np.array([[i] for i in range(0, 25)])
mat = increaseDimensionMatrix(1, 2)
new_data = np.matmul(data, mat)
print data
print new_data

plt.plot(new_data[:,0], new_data[:,1])
plt.xlim((-25, 25))
plt.ylim((-25, 25))
plt.show()