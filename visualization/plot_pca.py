import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_pca_2d(points, color, principal_components, ax, line_color="black", point_size=5, point_line_width=1, line_width=1, line_length=0.2):
	num_points = len(points)
	coordinates = np.zeros((num_points, 2, 2))
	for i in range(len(points)):
		coordinates[i][0][0] = points[i][0]
		coordinates[i][0][1] = points[i][1]
		coordinates[i][1][0] = points[i][0] + (line_length * principal_components[i][0][0])
		coordinates[i][1][1] = points[i][1] + (line_length * principal_components[i][0][1])
	ax.scatter(points[:,0], points[:,1], c=color, cmap=plt.cm.Spectral, s=point_size**2, zorder=2, linewidth=point_line_width)
	lines = LineCollection(coordinates, color=line_color, linewidths=line_width)
	ax.add_collection(lines)

def plot_pca_3d(points, color, principal_components, ax, line_color="black", point_size=5, point_line_width=1, line_width=1, line_length=0.2):
	num_points = len(points)
	coordinates = np.zeros((num_points, 2, 3))
	for i in range(len(points)):
		coordinates[i][0][0] = points[i][0]
		coordinates[i][0][1] = points[i][1]
		coordinates[i][0][2] = points[i][2]
		coordinates[i][1][0] = points[i][0] + (line_length * principal_components[i][0][0])
		coordinates[i][1][1] = points[i][1] + (line_length * principal_components[i][0][1])
		coordinates[i][1][2] = points[i][2] + (line_length * principal_components[i][0][2])
	ax.scatter(points[:,0], points[:,1], points[:,2], c=color, cmap=plt.cm.Spectral, s=point_size**2, zorder=2, linewidth=point_line_width)
	lines = Line3DCollection(coordinates, color=line_color, linewidths=line_width)
	ax.add_collection(lines)