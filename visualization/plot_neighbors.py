import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_neighbors_2d(points, color, neighbors_graph, ax, line_color="grey", point_size=5, line_width=1, edge_thickness=1):
	# See https://stackoverflow.com/questions/50040310/efficient-way-to-connect-the-k-nearest-neighbors-in-a-scatterplot-using-matplotl/50040839
	num_points = len(points)
	num_neighbors = np.count_nonzero(neighbors_graph[0].todense())
	coordinates = np.zeros((num_points, num_neighbors, 2, 2))
	for point_idx in range(num_points):
		for neighbor_idx in range(num_neighbors):
			target_idx = (np.where(neighbors_graph[point_idx].toarray()[0])[0])[neighbor_idx] # Evil numpy f***ery because sparse matrices behave strangely
			coordinates[point_idx, neighbor_idx, :, 0] = np.array([points[point_idx, :][0], points[target_idx, :][0]])
			coordinates[point_idx, neighbor_idx, :, 1] = np.array([points[point_idx, :][1], points[target_idx, :][1]])
	lines = LineCollection(coordinates.reshape((num_points*num_neighbors, 2, 2)), color=line_color, zorder=1, linewidths=edge_thickness)
	ax.scatter(points[:,0], points[:,1], c=color, cmap=plt.cm.Spectral, s=point_size**2, zorder=2, linewidth=line_width)
	ax.add_collection(lines)

def plot_neighbors_3d(points, color, neighbors_graph, ax, line_color="grey", point_size=5, line_width=1, edge_thickness=1):
	num_points = len(points)
	num_neighbors = np.count_nonzero(neighbors_graph[0].todense())
	coordinates = np.zeros((num_points, num_neighbors, 2, 3))
	for point_idx in range(num_points):
		for neighbor_idx in range(num_neighbors):
			target_idx = (np.where(neighbors_graph[point_idx].toarray()[0])[0])[neighbor_idx] # Evil numpy f***ery because sparse matrices behave strangely
			coordinates[point_idx, neighbor_idx, :, 0] = np.array([points[point_idx, :][0], points[target_idx, :][0]])
			coordinates[point_idx, neighbor_idx, :, 1] = np.array([points[point_idx, :][1], points[target_idx, :][1]])
			coordinates[point_idx, neighbor_idx, :, 2] = np.array([points[point_idx, :][2], points[target_idx, :][2]])
	lines = Line3DCollection(coordinates.reshape((num_points*num_neighbors, 2, 3)), color=line_color, zorder=1, linewidths=edge_thickness)
	ax.scatter(points[:,0], points[:,1], points[:,2], c=color, cmap=plt.cm.Spectral, s=point_size**2, zorder=2, linewidth=line_width)
	ax.add_collection(lines)

if __name__ == "__main__":
	from sklearn.neighbors import kneighbors_graph
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.pyplot as plt

	def test2d():
		fig, ax = plt.subplots()
		points = np.random.random(500).reshape((250, 2))
		color = np.random.random(250)
		graph = kneighbors_graph(points, 4, mode="distance", n_jobs=-1)
		plot_neighbors_2d(points, color, graph, ax)
		plt.show()

	def test3d():
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		points = np.random.random(750).reshape((250, 3))
		color = np.random.random(250)
		graph = kneighbors_graph(points, 4, mode="distance", n_jobs=-1)
		plot_neighbors_3d(points, color, graph, ax)
		plt.show()

	test2d()
	test3d()