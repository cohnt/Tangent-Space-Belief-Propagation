import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy
from utils import write, flush

num_iters = 200     # Number of iterations of the message passing algorithm to run
neighbors_k = 12    # The value of 'k' used for k-nearest-neighbors
num_points = 500    # Number of data points
data_noise = 0.0001 # How much noise is added to the data
num_samples = 200   # Numbers of samples used in the belief propagation algorithm
explore_perc = 0.1  # Fraction of uniform samples to keep exploring
initial_dim = 2     # The dimensionality of the incoming dataset (see "Load Dataset" below)
target_dim = 1      # The number of dimensions the data is being reduced to

output_dir = "results/"

write("\n")

################
# Load Dataset #
################
from datasets.dim_2.arc_curve import make_arc_curve

write("Generating dataset...")
flush()
t0 = time.time()
points, color = make_arc_curve(num_points, data_noise)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

#######################
# k-Nearest-Neighbors #
#######################
from sklearn.neighbors import kneighbors_graph
from visualization.plot_neighbors import plot_neighbors_2d

write("Computing nearest neighbors...")
flush()
t0 = time.time()
neighbor_graph = kneighbors_graph(points, neighbors_k, mode="distance", n_jobs=-1)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()
# NOTE: this is not a symmetric matrix.

write("Saving nearest neighbors plot...")
flush()
t0 = time.time()
fig, ax = plt.subplots()
plot_neighbors_2d(points, color, neighbor_graph, ax, point_size=3, line_width=0.25, edge_thickness=0.25, show_labels=True)
plt.savefig(output_dir + "nearest_neighbors.svg")
plt.close(fig)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

####################
# Initialize Graph #
####################
from utils import sparseMatrixToDict, sparseMaximum

# Make the matrix symmetric, and convert it to a dictionary
write("Initializing graph data structures...")
flush()
t0 = time.time()
neighbor_graph = sparseMaximum(neighbor_graph, neighbor_graph.T)
neighbor_dict = sparseMatrixToDict(neighbor_graph)
neighbor_pair_list = [(key, value) for key, arr in neighbor_dict.items() for value in arr]
num_messages = len(neighbor_pair_list)
# neighbor_pair_list represents the identification of the messages, i.e., "message 0" is at index 0
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()
