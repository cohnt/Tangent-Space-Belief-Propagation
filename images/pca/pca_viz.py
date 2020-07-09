import numpy as np
import matplotlib.pyplot as plt

mean = np.array([2, 2])
cov = 0.5 * np.array([[1, 0.9], [0.9, 1]])
x, y = np.random.multivariate_normal(mean, cov, 75).T

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(np.array([x,y]).T)
colors = pca.transform(np.array([x,y]).T)[:,0]
colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))

major_axis = pca.components_[0]
minor_axis = pca.components_[1]

plt.figure(figsize=(5,5))
for i in range(len(x)):
	px = x[i]
	py = y[i]
	proj = pca.transform([[px, py]])
	point2 = (proj[0,0] * major_axis) + pca.mean_
	plt.plot([px, point2[0]], [py, point2[1]], color="blue", zorder=-1)
plt.scatter(x, y, c=colors, cmap=plt.cm.Spectral, s=7**2, edgecolors="black", zorder=2)
plt.plot([pca.mean_[0]+3*major_axis[0], pca.mean_[0]-3*major_axis[0]], [pca.mean_[1]+3*major_axis[1], pca.mean_[1]-3*major_axis[1]], color="black", linewidth=2, zorder=1)
plt.plot([pca.mean_[0]+minor_axis[0], pca.mean_[0]-minor_axis[0]], [pca.mean_[1]+minor_axis[1], pca.mean_[1]-minor_axis[1]], color="black", linewidth=2, zorder=0)
plt.xlim((0,4))
plt.ylim((0,4))
plt.show()

plt.figure(figsize=(5,5))
plt.plot([pca.mean_[0]+3*major_axis[0], pca.mean_[0]-3*major_axis[0]], [pca.mean_[1]+3*major_axis[1], pca.mean_[1]-3*major_axis[1]], color="black", linewidth=2, zorder=1)
plt.plot([pca.mean_[0]+minor_axis[0], pca.mean_[0]-minor_axis[0]], [pca.mean_[1]+minor_axis[1], pca.mean_[1]-minor_axis[1]], color="black", linewidth=2, zorder=0)
points = []
for i in range(len(x)):
	px = x[i]
	py = y[i]
	proj = pca.transform([[px, py]])
	point2 = (proj[0,0] * major_axis) + pca.mean_
	points.append(point2)
points = np.array(points)
plt.scatter(points[:,0], points[:,1], c=colors, cmap=plt.cm.Spectral, zorder=5, s=7**2, edgecolors="black")
plt.xlim((0,4))
plt.ylim((0,4))
plt.show()
