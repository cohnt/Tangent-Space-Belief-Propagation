import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy
from utils import write, flush

num_iters = 100     # Number of iterations of the message passing algorithm to run
neighbors_k = 12    # The value of 'k' used for k-nearest-neighbors
num_points = 500    # Number of data points
data_noise = 0.0001 # How much noise is added to the data
num_samples = 25   # Numbers of samples used in the belief propagation algorithm
explore_perc = 0.1  # Fraction of uniform samples to keep exploring
source_dim = 2      # The dimensionality of the incoming dataset (see "Load Dataset" below)
target_dim = 1      # The number of dimensions the data is being reduced to

message_resample_cov = np.eye(target_dim) * 0.01 # TODO: Change

output_dir = "results/"

write("\n")
################
# Load Dataset #
################
from datasets.dim_2.arc_curve import make_arc_curve
# from datasets.dim_2.s_curve import make_s_curve
# from datasets.dim_2.o_curve import make_o_curve

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
# neighbor_graph is stored as a sparse matrix.
# Note that neighbor_graph is not necessarily symmetric, such as the case where point x
# is a nearest neighbor of point y, but point y is *not* a nearest neighbor of point x.
# We fix this later on...

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

write("Initializing graph data structures...")
flush()
t0 = time.time()
# Make the matrix symmetric by taking max(G, G^T)
neighbor_graph = sparseMaximum(neighbor_graph, neighbor_graph.T)
# This dictionary will have the structure point_idx: [list, of, neighbor_idx]
neighbor_dict = sparseMatrixToDict(neighbor_graph)
# This extracts all pairs of neighbors from the dictionary and stores them as a list of tuples.
# neighbor_pair_list represents the identification of the messages, i.e., "message 0" is
# so defined by being at index 0 of neighbor_pair_list.
neighbor_pair_list = [(key, value) for key, arr in neighbor_dict.items() for value in arr]
num_messages = len(neighbor_pair_list)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

###############
# Measure PCA #
###############
from sklearn.decomposition import PCA
from visualization.plot_pca import plot_pca_2d

# n_components is the number of principal components pca will compute
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
from utils import randomBasis

class Message():
	def __init__(self, num_samples, source_dim, target_dim):
		# If num_samples=N and source_dim=n, and target_dim=m, then:
		# self.ts is a list of ordered bases of m-dimensional (i.e. spanned by m
		# unit vectors) subspaces in R^n, so it's of shape (N, m, n)
		# self.weights is a list of weights, so it's of shape (N)
		self.ts = np.zeros((num_samples, target_dim, source_dim))
		self.weights = np.zeros(num_samples)

def randomTangentSpace(num_samples, source_dim, target_dim):
	# Return a random list of size target_dim orthonormal vectors in source_dim.
	# This represents the basis of a random subspace of dimension target_dim in
	# the higher dimensional space of dimension source_dim
	ts = np.zeros((num_samples, target_dim, source_dim))
	for i in range(num_samples):
		ts[i][:] = randomBasis(source_dim)[0:target_dim];
	return ts

# This initializes messages_prev and messages_next as num_points by num_points arrays of Nones.
# Where appropriate, the Nones will be replaced by Message objects
messages_prev = [[None for __ in range(num_points)] for _ in range(num_points)]
messages_next = [[None for __ in range(num_points)] for _ in range(num_points)]
for key, value in neighbor_pair_list:
	# Note that key represents where the message is coming from and value represents where the message is going to
	# In other words, messages[key][value] === m_key->value
	messages_prev[key][value] = Message(num_samples, source_dim, target_dim)
	messages_prev[key][value].ts = randomTangentSpace(num_samples, source_dim, target_dim, observations[key])
	messages_prev[key][value].weights = np.zeros(num_samples) + (1.0 / num_samples) # Evenly weight each sample for now

	# We don't initialize any values into messages_next
	messages_next[key][value] = Message(num_samples, source_dim, target_dim)

###################
# Message Passing #
###################
from utils import weightedSample, list_mvn, projSubspace
from scipy.linalg import subspace_angles

class Belief():
	def __init__(self, num_samples, source_dim, target_dim):
		# If num_samples=N and source_dim=n, and target_dim=m, then:
		# self.ts is a list of ordered bases of m-dimensional (i.e. spanned by m
		# unit vectors) subspaces in R^n, so it's of shape (N, m, n)
		# self.weights is a list of weights, so it's of shape (N)
		self.ts = np.zeros((num_samples, target_dim, source_dim))
		self.weights = np.zeros(num_samples)

belief = [Belief(num_samples, source_dim, target_dim) for _ in range(num_points)]

def weightMessage(m_next, m_prev, neighbor, current):
	t = neighbor
	s = current
	# Weight m_t->s

	# We will compute the unary and prior weights separately, then multiply them together
	weights_unary = np.zeros(num_samples)
	weights_prior = np.zeros(num_samples)
	weights = np.zeros(num_samples)

	# Get all of the neighbors of t, but don't include s in the list.
	neighbors = neighbor_dict[t][:]
	neighbors.remove(s)
	num_neighbors = len(neighbors)

	for i in range(num_samples):
		ts_s = m_next[t][s].ts[i]
		ts_t = sampleNeighbor(pos_s, ts_s, t, s)

		weights_unary[i] = weightUnary(pos_t, ts_t, t)

		if num_neighbors > 0:
			# Since we're doing k-nearest neighbors, this is always true. But if we used another
			# neighbor-finding algorithm, this might not necessarily be true. In theory, this would
			# still work even if num_neighbors were 0, since np.prod of an empty list returns 1.0.
			weights_from_priors = np.zeros(num_neighbors)
			for j in range(num_neighbors):
				u = neighbors[j]
				weights_from_priors[j] = weightPrior(ts_s, m_prev, u, t, s)
			weights_prior[i] = np.prod(weights_from_priors)

	# Finally, we normalize the weights. We have to check that we don't have all weights zero. This
	# shouldn't ever happen, but the check is here just in case.
	max_weight_unary = np.max(weights_unary)
	max_weight_prior = np.max(weights_prior)

	if max_weight_unary != 0:
		weights_unary = weights_unary / max_weight_unary
	else:
		# All of weights_unary is 0 (since negative weights aren't possible). To make
		# every element 1/num_samples, we just add that value and numpy does the rest.
		weights_unary = weights_unary + (1.0 / num_samples)

	if max_weight_prior != 0:
		weights_prior = weights_prior / max_weight_prior
	else:
		# All of weights_prior is 0 (since negative weights aren't possible). To make
		# every element 1/num_samples, we just add that value and numpy does the rest.
		weights_prior = weights_prior + (1.0 / num_samples)

	# np.multiply does an element-by-element multiplication.
	weights = np.multiply(weights_unary, weights_prior)
	return weights

def sampleNeighbor(ts, neighbor, current):
	# We assume neighbors have very similar orientation
	return ts[:]

def weightUnary(ts, idx):
	# We compute the principal angles between the observation (via PCA on the
	# neighborhood), and our prediction. We use scipy.linalg.subspace_angles. This
	# function expects that the inputs will be matrices, with the basis vectors as
	# columns. ts is stored as a list of basis vectors (read: rows), so we have
	# to transpose it. The same is true for the observation
	principal_angles = subspace_angles(ts.transpose(), observations[idx].transpose())
	total_angle_error = np.sum(principal_angles)
	weight = 1.0 / (1.0 + total_angle_error)

	return weight

def weightPrior(ts_s, m_prev, neighbor_neighbor, neighbor, current):
	u = neighbor_neighbor
	t = neighbor
	s = current
	# Use m_u->t to help weight m_t->s. Really, we're just worried about weighting
	# a given sample right now, based on m_u->t from a previous iteration.
	weight_prior = 0.0
	for i in range(num_samples):
		ts_t = m_prev[u][t].ts[i]
		# dist2 = (np.asarray(pos, dtype=float) - np.asarray(pos_pred, dtype=float)) ** 2
		# weight_pairwise = 1/(1+dist2)
		
		# We have a relation between the orientations of adjacent nodes -- they should be similar
		principal_angles = subspace_angles(ts_s.transpose(), ts_t.transpose())
		weight = 1.0 / (1.0 + np.sum(principal_angles))

		weight_prior = weight_prior + (m_prev[u][t].weights[i] * weight)
	return weight_prior