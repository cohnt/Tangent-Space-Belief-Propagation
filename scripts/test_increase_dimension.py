import numpy as np
import matplotlib.pyplot as plt

from utils import increaseDimension

data = np.array([[i] for i in range(0, 25)])
new_data = increaseDimension(data, 2)
print data
print new_data

plt.plot(new_data[:,0], new_data[:,1])
plt.xlim((-25, 25))
plt.ylim((-25, 25))
plt.show()