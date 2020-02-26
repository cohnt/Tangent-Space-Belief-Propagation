import numpy as np
import matplotlib.pyplot as plt

from plot_neighbors import plot_neighbors_2d

def unrollAnimation(points, color, true_parameters, neighbor_graph, base_filename, num_images=50, delay=10, line_color="grey", point_size=5, line_width=1, edge_thickness=1, show_labels=False):
	points_by_frame = unrollGetPoints(points, true_parameters, num_images)
	for i in range(len(points_by_frame)):
		point_set = points_by_frame[i]
		fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
		plot_neighbors_2d(point_set, color, neighbor_graph, ax, point_size=point_size, line_width=line_width, edge_thickness=edge_thickness, show_labels=False)
		min_x = np.min([np.min(true_parameters[:,0]), np.min(points[:,0])])
		max_x = np.max([np.max(true_parameters[:,0]), np.max(points[:,0])])
		min_y = np.min([0, np.min(points[:,1])])
		max_y = np.max([0, np.max(points[:,1])])
		ax.set_xlim((min_x, max_x))
		ax.set_ylim((min_y, max_y))
		plt.savefig(base_filename + "%03d" % i + ".png")
		plt.close(fig)

def unrollGetPoints(points, true_parameters, num_images):
	x_vals, y_vals = np.transpose(points)
	goal_x_vals = true_parameters[:,0]
	goal_y_vals = np.zeros(len(true_parameters))
	points_by_frame = []
	num_points = len(points)
	for i in range(0, num_images):
		points_by_frame.append(points.copy())
		for j in range(len(points)):
			x_min = x_vals[j]
			x_max = goal_x_vals[j]
			y_min = y_vals[j]
			y_max = goal_y_vals[j]
			t = float(i) / float(num_images-1)
			x = x_min + (t * (x_max - x_min))
			y = y_min + (t * (y_max - y_min))
			points_by_frame[i][j,:] = [x, y]
	return points_by_frame