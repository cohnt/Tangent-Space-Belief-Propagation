import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import time
import sys

num_iters = 5      # Number of iterations of the message passing algorithm to run
neighbors_k = 8    # The value of 'k' used for k-nearest-neighbors
num_points = 500   # Number of data points
data_noise = 0.001 # How much noise is added to the data
num_samples = 200  # Numbers of samples used in the belief propagation algorithm

output_dir = "results/"

sys.stdout.write("\n")

################
# Load Dataset #
################
from datasets.dim_2.s_curve import make_s_curve

sys.stdout.write("Generating dataset...")
sys.stdout.flush()
t0 = time.time()
points, color = make_s_curve(num_points, data_noise)
t1 = time.time()
sys.stdout.write("Done! dt=%f\n" % (t1-t0))
sys.stdout.flush()

#######################
# k-Nearest-Neighbors #
#######################
from sklearn.neighbors import kneighbors_graph
from visualization.plot_neighbors import plot_neighbors_2d

sys.stdout.write("Computing nearest neighbors...")
sys.stdout.flush()
t0 = time.time()
neighbor_graph = kneighbors_graph(points, neighbors_k, mode="distance", n_jobs=-1)
t1 = time.time()
sys.stdout.write("Done! dt=%f\n" % (t1-t0))
sys.stdout.flush()
# NOTE: this is not a symmetric matrix.

sys.stdout.write("Saving nearest neighbors plot...")
sys.stdout.flush()
t0 = time.time()
fig, ax = plt.subplots()
plot_neighbors_2d(points, color, neighbor_graph, ax, point_size=3, line_width=0.25, edge_thickness=0.25)
plt.savefig(output_dir + "nearest_neighbors.svg")
plt.close(fig)
t1 = time.time()
sys.stdout.write("Done! dt=%f\n" % (t1-t0))
sys.stdout.flush()

####################
# Initialize Graph #
####################
from utils import sparseMatrixToDict, sparseMaximum

# Make the matrix symmetric, and convert it to a dictionary
sys.stdout.write("Initializing graph data structures...")
sys.stdout.flush()
t0 = time.time()
neighbor_graph = sparseMaximum(neighbor_graph, neighbor_graph.T)
neighbor_dict = sparseMatrixToDict(neighbor_graph)
neighbor_pair_list = [(key, value) for key, arr in neighbor_dict.items() for value in arr]
num_messages = len(neighbor_pair_list)
t1 = time.time()
sys.stdout.write("Done! dt=%f\n" % (t1-t0))
sys.stdout.flush()

#######################
# Initialize Messages #
#######################

message_list_shape = (num_points, num_points)

###################
# Message Passing #
###################

for iter_num in range(num_iters):
	##################
	# Message Update #
	##################

	pass # TODO

	#################
	# Belief Update #
	#################

	pass # TODO

sys.stdout.write("\n")