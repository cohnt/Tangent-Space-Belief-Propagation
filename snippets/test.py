import numpy as np
import matplotlib.pyplot as plt
import time

from datasets.dim_2.o_curve import make_o_curve
from sklearn.neighbors import kneighbors_graph
from visualization.plot_neighbors import plot_neighbors_2d
from sklearn.decomposition import PCA
from utils import sparseMaximum

neighbors_k = 12
num_points = 400

pca = PCA(n_components=1)

points, color = make_o_curve(num_points, 0.0001)
neighbor_graph = kneighbors_graph(points, neighbors_k, mode="distance", n_jobs=-1)
neighbor_graph = sparseMaximum(neighbor_graph, neighbor_graph.T)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
plot_neighbors_2d(points, color, neighbor_graph, ax1)

for i in range(num_points):
	og_point = points[i]
	row = neighbor_graph.toarray()[i]
	neighbors = np.nonzero(row)[0]
	neighborhood = points[neighbors]
	pca.fit(neighborhood)
	vec1 = pca.components_[0]
	for j in range(len(neighborhood)):
		point = neighborhood[j]
		if not np.array_equal(og_point, point):
			vec2 = point - og_point
			val = np.arccos(np.abs(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))) * 180 / np.pi
			if val >= 45.0:
				neighbor_graph[i, neighbors[j]] = 0
				neighbor_graph[neighbors[j], i] = 0
