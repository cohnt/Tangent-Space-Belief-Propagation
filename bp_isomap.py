import numpy as np
import matplotlib.pyplot as plt

num_iters = 5      # Number of iterations of the message passing algorithm to run
neighbors_k = 8   # The value of 'k' used for k-nearest-neighbors
num_points = 500   # Number of data points
data_noise = 0.001 # How much noise is added to the data

output_dir = "results/"

################
# Load Dataset #
################
from datasets.dim_2.s_curve import make_s_curve

points, color = make_s_curve(num_points, data_noise)

#######################
# k-Nearest-Neighbors #
#######################
from sklearn.neighbors import kneighbors_graph
from visualization.plot_neighbors import plot_neighbors_2d

neighbor_graph = kneighbors_graph(points, neighbors_k, mode="distance", n_jobs=-1)
# NOTE: this is not a symmetric matrix.

fig, ax = plt.subplots()
plot_neighbors_2d(points, color, neighbor_graph, ax, point_size=3, line_width=0.25, edge_thickness=0.25)
plt.savefig(output_dir + "nearest_neighbors.svg")
plt.close(fig)

####################
# Initialize Graph #
####################
from utils import sparseMatrixToDict, sparseMaximum

# Make the matrix symmetric, and convert it to a dictionary
neighbor_graph = sparseMaximum(neighbor_graph, neighbor_graph.T)
neighbor_dict = sparseMatrixToDict(neighbor_graph)

#######################
# Initialize Messages #
#######################

# TODO

###################
# Message Passing #
###################

for iter_num in range(num_iters):
	pass
	# TODO