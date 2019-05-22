import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import time
import sys
import copy

num_iters = 5      # Number of iterations of the message passing algorithm to run
neighbors_k = 8    # The value of 'k' used for k-nearest-neighbors
num_points = 500   # Number of data points
data_noise = 0.001 # How much noise is added to the data
num_samples = 200  # Numbers of samples used in the belief propagation algorithm

initial_dim = 2    # The dimensionality of the incoming dataset (see "Load Dataset" below)
target_dim = 1     # The number of dimensions the data is being reduced to

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

class Message:
	def __init__(self):
		self.pos = np.zeros((num_samples, target_dim))
		self.weights = np.zeros(num_samples)

messages_prev = [[None for j in range(num_points)] for i in range(num_points)]
messages_next = [[None for j in range(num_points)] for i in range(num_points)]
for key, value in neighbor_pair_list:
	# Note that key represents where the message is coming from and value represents where the message is going to
	# In other words, messages[key][value] === m_key->value
	messages_prev[key][value] = Message()
	messages_prev[key][value].pos = np.random.uniform(-2, 2, (num_samples, target_dim))
	messages_prev[key][value].weights = np.zeros(num_samples) + (1.0 / num_samples)

	messages_next[key][value] = Message()
	# We don't initialize any values into messages_next

###################
# Message Passing #
###################

class Belief:
	def __init__(self):
		self.pos = np.zeros((num_samples, target_dim))
		self.weights = np.zeros(num_samples)

belief = [Belief() for _ in range(num_points)]

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