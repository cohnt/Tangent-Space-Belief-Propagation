import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

t = np.arange(0, 1, 0.01)
X = np.array([3*t-5, -2*t+4]).transpose()

pca = PCA(n_components=1)
y = pca.fit_transform(X)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
ax1.scatter(X[:,0], X[:,1], c=t, cmap=plt.cm.Spectral)
ax1.set_title("Input Space")

ax2.scatter(y, np.ones(len(y)), c=t, cmap=plt.cm.Spectral)
ax2.set_title("Feature Space")

plt.show()