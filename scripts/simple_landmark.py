import numpy as np
import matplotlib.pyplot as plt

neighbors_k = 8

# landmark_coords = [
# 	np.array([0., 0.]),
# 	np.array([5., 5.]),
# 	np.array([0.,10.])
# ]
# num_landmarks = len(landmark_coords)
landmark_coords = []
num_landmarks = 3
for _ in range(num_landmarks):
	landmark_coords.append(np.random.uniform(low=0.0, high=10.0, size=(2)))

print "Landmarks:"
for landmark in landmark_coords:
	print landmark

# Build up a list of points to measure at
x_vals = np.linspace(0, 10, num=21)
y_vals = np.linspace(0, 10, num=21)
xx, yy = np.meshgrid(x_vals, y_vals)
points = np.stack((np.ravel(xx), np.ravel(yy)), axis=-1)
num_points = len(points)

range_data = np.zeros((num_points, num_landmarks))

def noise():
	# return np.random.uniform(low=-0.5, high=0.5)
	return np.random.normal(loc=0.0, scale=0.5)

for i in range(num_points):
	for j in range(num_landmarks):
		range_data[i][j] = np.linalg.norm(points[i] - landmark_coords[j]) + noise()

from sklearn.manifold import LocallyLinearEmbedding, MDS, Isomap, SpectralEmbedding, TSNE

method = Isomap(n_neighbors=neighbors_k, n_components=2, n_jobs=-1)
feature_coords = method.fit_transform(range_data)

fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].scatter(feature_coords[:,0], feature_coords[:,1], c=points[:,0]/10.0, cmap=plt.cm.Spectral)
axes[1].scatter(feature_coords[:,0], feature_coords[:,1], c=points[:,1]/10.0, cmap=plt.cm.Spectral)
plt.show()

from utils import pairwiseDistErr
from visualization.error_plots import regressionErrorCharacteristic, listRegressionErrorCharacteristic

# max_err = pairwiseDistErr(feature_coords, points, dist_metric="l2", mat_norm="max")
# print "Maximum error: %f" % max_err

# fig, ax = plt.subplots()
# regressionErrorCharacteristic(ax, feature_coords, points, dist_metric="l2")
# ax.set_title("Regression Error Characteristic")
# plt.show()

from sklearn.neighbors import kneighbors_graph
from visualization.plot_neighbors import plot_neighbors_2d
from utils import sparseMatrixToDict, sparseMaximum

neighbor_graph = kneighbors_graph(range_data, neighbors_k, mode="distance", n_jobs=-1)
neighbor_graph = sparseMaximum(neighbor_graph, neighbor_graph.T)
neighbor_dict = sparseMatrixToDict(neighbor_graph)
neighbor_pair_list = [(key, value) for key, arr in neighbor_dict.items() for value in arr]

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
plot_neighbors_2d(points, points[:,0]/10.0, neighbor_graph, ax, show_labels=False)
ax.set_title("Nearest Neighbors (k=%d)\n" % neighbors_k)
plt.show()