import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy
from utils import write, flush

num_iters = 200     # Number of iterations of the message passing algorithm to run
neighbors_k = 12    # The value of 'k' used for k-nearest-neighbors
num_points = 500    # Number of data points
data_noise = 0.00001 # How much noise is added to the data
num_samples = 200   # Numbers of samples used in the belief propagation algorithm
explore_perc = 0.1  # Fraction of uniform samples to keep exploring
source_dim = 2      # The dimensionality of the incoming dataset (see "Load Dataset" below)
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
plot_neighbors_2d(points, color, neighbor_graph, ax, point_size=1, line_width=0.1, edge_thickness=0.1, show_labels=False)
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

###############
# Measure PCA #
###############
from sklearn.decomposition import PCA
from visualization.plot_pca import plot_pca_2d

pca = PCA(n_components=target_dim)
observations = [None for i in range(num_points)]

write("Computing PCA observations...")
flush()
t0 = time.time()
for i in range(num_points):
	og_point = points[i]
	row = neighbor_graph.toarray()[i]
	neighbors = np.nonzero(row)[0]
	neighborhood = points[neighbors]
	pca.fit(neighborhood)
	# vec1 = pca.components_[0]
	observations[i] = pca.components_[0:target_dim]
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

write("Saving PCA observations plot...")
flush()
t0 = time.time()
fig, ax = plt.subplots()
plot_pca_2d(points, color, observations, ax, point_size=1, point_line_width=0.1, line_width=0.1, line_length=0.05)
plt.savefig(output_dir + "pca_observations.svg")
plt.close(fig)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

#######################
# Initialize Messages #
#######################
class Message():
	def __init__(self, num_samples, source_dim, target_dim):
		self.pos = np.zeros((num_samples, target_dim))
		self.orien = np.zeros((num_samples, target_dim, source_dim))
		self.weights = np.zeros(num_samples)

def randomPos(num_samples, target_dim):
	return np.random.uniform(-2, 2, (num_samples, target_dim))

def randomOrien(num_samples, source_dim, target_dim, observed_orien):
	return np.tile(observed_orien, (num_samples, target_dim, 1))

messages_prev = [[None for __ in range(num_points)] for _ in range(num_points)]
messages_next = [[None for __ in range(num_points)] for _ in range(num_points)]
for key, value in neighbor_pair_list:
	# Note that key represents where the message is coming from and value represents where the message is going to
	# In other words, messages[key][value] === m_key->value
	messages_prev[key][value] = Message(num_samples, source_dim, target_dim)
	messages_prev[key][value].pos = randomPos(num_samples, target_dim)
	messages_prev[key][value].orien = randomOrien(num_samples, source_dim, target_dim, observations[key])
	messages_prev[key][value].weights = np.zeros(num_samples) + (1.0 / num_samples)

	messages_next[key][value] = Message(num_samples, source_dim, target_dim)
	# We don't initialize any values into messages_next

###################
# Message Passing #
###################
from utils import weightedSample, list_mvn

class Belief():
	def __init__(self, num_samples, source_dim, target_dim):
		self.pos = np.zeros((num_samples, target_dim))
		self.orien = np.zeros((num_samples, target_dim, source_dim))
		self.weights = np.zeros(num_samples)

belief = [Belief(num_samples, source_dim, target_dim) for _ in range(num_points)]

for iter_num in range(1, num_iters+1):
	write("\nIteration %d\n" % iter_num)

	message_time = 0
	belief_time = 0
	image_time = 0
	total_time = 0

	##################
	# Message Update #
	##################
	write("Performing message update...")
	flush()
	t0 = time.time()

	pass # TODO

	t1 = time.time()
	message_time = t1-t0
	write("Done! dt=%f\n" % message_time)

	#################
	# Belief Update #
	#################
	write("Performing belief update...")
	flush()
	t0 = time.time()

	pass # TODO

	t1 = time.time()
	belief_time = t1-t0
	write("Done! dt=%f\n" % belief_time)

	################
	# Write Images #
	################
	write("Writing images...")
	flush()
	t0 = time.time()

	pass # TODO

	t1 = time.time()
	image_time = t1-t0
	write("Done! dt=%f\n" % image_time)

	total_time = message_time + belief_time + image_time
	write("Total iteration time: %f\n" % total_time)