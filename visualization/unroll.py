import numpy as np
import matplotlib.plt as pyplot

from plot_neighbors import plot_neighbors_2d

def unroll(points, color, true_parameters, neighborhood_graph, num_images=50, line_color="grey", point_size=5, line_width=1, edge_thickness=1, show_labels=False):
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
	# print points_by_frame