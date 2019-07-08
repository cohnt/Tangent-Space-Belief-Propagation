import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from utils import Belief

def plot_reconstruction_1d_2d(points, order, ax):
	num_points = len(points)
	coordinates = np.zeros((num_points-1, 2, 2))
	for i in range(len(points)-1):
		coordinates[i][0][0] = points[order[i]][0]
		coordinates[i][0][1] = points[order[i]][1]
		coordinates[i][1][0] = points[order[i+1]][0]
		coordinates[i][1][1] = points[order[i+1]][1]
	lines = LineCollection(coordinates, color="red", linewidths=5)
	ax.add_collection(lines)