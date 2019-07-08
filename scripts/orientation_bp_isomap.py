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
# from datasets.dim_2.arc_curve import make_arc_curve
# # from datasets.dim_2.s_curve import make_s_curve
from datasets.dim_2.o_curve import make_o_curve

write("Generating dataset...")
flush()
t0 = time.time()
points, color = make_o_curve(num_points, data_noise)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

# write("Generating dataset...")
# flush()
# t0 = time.time()
# points = np.array([[0, 0], [1, 1], [2, 2.5], [3, 4], [4, 4.5], [5, 7]])
# color = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# t1 = time.time()
# write("Done! dt=%f\n" % (t1-t0))
# flush()

# num_points = 6
# neighbors_k = 3

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
class Message():
	def __init__(self, num_samples, source_dim, target_dim):
		# If num_samples=N, source_dim=n, and target_dim=m, then:
		# self.pos is a list of N points in R^m, so it's of shape (N, m)
		# self.orien is a list of ordered bases of m-dimensional (i.e. spanned by m
		# unit vectors) subspaces in R^n, so it's of shape (N, m, n)
		# self.weights is a list of weights, so it's of shape (N)
		self.pos = np.zeros((num_samples, target_dim))
		self.orien = np.zeros((num_samples, target_dim, source_dim))
		self.weights = np.zeros(num_samples)

def randomPos(num_samples, target_dim):
	# Our current dataset has a length of pi/2=1.57, so no matter which point is "anchored" at
	# 0, the furthest point will definitely be somewhere in the interval (-2, 2).
	# The shape of the output matches the dimension of a Message or Belief
	return np.random.uniform(-2, 2, (num_samples, target_dim))

def randomOrien(num_samples, source_dim, target_dim, observed_orien):
	# For now, for the message m_t->s, we expect s to have the same orientation as t. This will
	# almost certainly be changed sometime in the future.
	# np.tile(vec, (a, b, 1)) creates an array of shape(a, b, len(vec)), so the output shape
	# matches the definition of a Message or Belief. For more details, see
	# https://stackoverflow.com/questions/22634265/python-concatenate-or-clone-a-numpy-array-n-times
	return np.tile(observed_orien, (num_samples, target_dim, 1))

# This initializes messages_prev and messages_next as num_points by num_points arrays of Nones.
# Where appropriate, the Nones will be replaced by Message objects
messages_prev = [[None for __ in range(num_points)] for _ in range(num_points)]
messages_next = [[None for __ in range(num_points)] for _ in range(num_points)]
for key, value in neighbor_pair_list:
	# Note that key represents where the message is coming from and value represents where the message is going to
	# In other words, messages[key][value] === m_key->value
	messages_prev[key][value] = Message(num_samples, source_dim, target_dim)
	messages_prev[key][value].pos = randomPos(num_samples, target_dim)
	messages_prev[key][value].orien = randomOrien(num_samples, source_dim, target_dim, observations[key])
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
		# If num_samples=N, source_dim=n, and target_dim=m, then:
		# self.pos is a list of N points in R^m, so it's of shape (N, m)
		# self.orien is a list of ordered bases of m-dimensional (i.e. spanned by m
		# unit vectors) subspaces in R^n, so it's of shape (N, m, n)
		# self.weights is a list of weights, so it's of shape (N)
		self.pos = np.zeros((num_samples, target_dim))
		self.orien = np.zeros((num_samples, target_dim, source_dim))
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
		pos_s = m_next[t][s].pos[i]
		orien_s = m_next[t][s].orien[i]
		pos_t, orien_t = sampleNeighbor(pos_s, orien_s, t, s)

		weights_unary[i] = weightUnary(pos_t, orien_t, t)

		if num_neighbors > 0:
			# Since we're doing k-nearest neighbors, this is always true. But if we used another
			# neighbor-finding algorithm, this might not necessarily be true. In theory, this would
			# still work even if num_neighbors were 0, since np.prod of an empty list returns 1.0.
			weights_from_priors = np.zeros(num_neighbors)
			for j in range(num_neighbors):
				u = neighbors[j]
				weights_from_priors[j] = weightPrior(pos_s, orien_s, m_prev, u, t, s)
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

def sampleNeighbor(pos, orien, neighbor, current):
	displacement_vec = points[neighbor] - points[current]
	# We now want to project this displacement vector onto the subspace spanned by
	# orien. Because we require orien to be orthonormal, we can use vector space
	# projection as defined by http://mathworld.wolfram.com/VectorSpaceProjection.html
	# For each basis_vec of orien, the component of displacement_vec along that component
	# is given by <basis_vec, displacement_vec>. We can then add that to the position of
	# the current node (pos) to get the predicted position of the neighbor
	individual_components = np.zeros(target_dim)
	for i in range(target_dim):
		basis_vec = orien[i]
		individual_components[i] = np.dot(basis_vec, displacement_vec)
	n_pos = pos + individual_components
	n_orien = orien
	return n_pos, n_orien

def weightUnary(pos, orien, idx):
	weight_unary = 1.0
	if idx == 0:
		# The point at index 0 is fixed at the origin of the projected space, so
		# that there will actually be a single ground truth (in theory).
		dist2 = np.sum(np.asarray(pos, dtype=float) ** 2)
		weight_unary = 1.0 / (1.0 + (10.0 * dist2))

	# Now, we compute the principal angles between the observation (via PCA on the
	# neighborhood), and our prediction. We use scipy.linalg.subspace_angles. This
	# function expects that the inputs will be matrices, with the basis vectors as
	# columns. orien is stored as a list of basis vectors (read: rows), so we have
	# to transpose it. The same is true for the observation
	principal_angles = subspace_angles(orien.transpose(), observations[idx].transpose())
	total_angle_error = np.sum(principal_angles)
	angle_weight = 1.0 / (1.0 + total_angle_error)

	weight_unary = weight_unary * angle_weight
	return weight_unary

def weightPrior(pos_s, orien_s, m_prev, neighbor_neighbor, neighbor, current):
	u = neighbor_neighbor
	t = neighbor
	s = current
	# Use m_u->t to help weight m_t->s. Really, we're just worried about weighting
	# a given sample right now, based on m_u->t from a previous iteration.
	weight_prior = 0.0
	for i in range(num_samples):
		pos_t = m_prev[u][t].pos[i]
		orien_t = m_prev[u][t].orien[i]
		# dist2 = (np.asarray(pos, dtype=float) - np.asarray(pos_pred, dtype=float)) ** 2
		# weight_pairwise = 1/(1+dist2)
		
		# We have a relation between the orientations of adjacent nodes -- they should be similar
		principal_angles = subspace_angles(orien_s.transpose(), orien_t.transpose())
		orien_weight = 1.0 / (1.0 + np.sum(principal_angles))

		# We have a relation between the orientations and predicted positions of adjacent nodes
		d_st = np.sum((pos_t - projSubspace(orien_t, points[s])) ** 2)
		d_ts = np.sum((pos_s - projSubspace(orien_s, points[t])) ** 2)
		distance_weight = 1.0 / (1.0 + ((d_st + d_ts) / 2.0))

		weight_4tuplewise = orien_weight * distance_weight

		weight_prior = weight_prior + (m_prev[u][t].weights[i] * weight_4tuplewise)
	return weight_prior

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

	if iter_num == 1:
		messages_next = copy.deepcopy(messages_prev)
	else:
		# Resample messages from previous belief using importance sampling. The key function used here
		# is weightedSample, from utils.py, uses stratified resampling, a resampling method common for
		# particle filters and similar probabilistic methods.
		for neighbor_pair in neighbor_pair_list:
			t = neighbor_pair[0]
			s = neighbor_pair[1]
			# We will update m_t->s

			start_ind = 0
			max_weight_ind = np.argmax(belief[s].weights)
			max_weight = belief[s].weights[max_weight_ind]
			if max_weight != 1.0 / num_samples:
				# Not all samples have the same weight, so we keep the highest weighted sample
				start_ind = 1

			# Note that we don't actually care about the weights we are assigning in this step, since
			# all the samples will be properly weighted later on. In theory, we don't have to assign
			# the samples weights at all, but it seems natural to at least give them the "default"
			# weight of 1.0 / num_samples.

			# If there is a best sample, keep it. In theory, this could be expanded to take the best
			# n samples, with only a little modification. One could use argsort to sort the array, but
			# this would be a pretty big performance hit. This stackoverflow answer suggests using
			# argpartition instead: https://stackoverflow.com/a/23734295/
			if start_ind == 1:
				messages_next[t][s].pos[0:start_ind] = belief[s].pos[max_weight_ind][:]
				messages_next[t][s].orien[0:start_ind] = belief[s].orien[max_weight_ind][:]
				messages_next[t][s].weights[0:start_ind] = 1.0 / num_samples

			# Some samples will be randomly seeded across the state space. Specifically, the
			# interval [start_ind, end_rand_in). This will be slightly less than explore_perc
			# if a maximum weight value is kept from the previous iteration.
			end_rand_ind = int(np.floor(num_samples * explore_perc))
			this_section_num_samples = end_rand_ind - start_ind
			# If explore_perc is so small (or just zero) that the number of random samples
			# is 0, then we don't need to do this step.
			if this_section_num_samples > 0:
				messages_next[t][s].pos[start_ind:end_rand_ind] = randomPos(this_section_num_samples, target_dim)
				messages_next[t][s].orien[start_ind:end_rand_ind] = randomOrien(this_section_num_samples, source_dim, target_dim, observations[t])
				messages_next[t][s].weights[start_ind:end_rand_ind] = 1.0 / num_samples

			# Finally, we generate the remaining samples (i.e. the interval [end_rand_in, num_samples))
			# by resampling from the belief of the previous iteration, with a little added noise.
			num_samples_left = num_samples - end_rand_ind
			belief_inds = weightedSample(belief[s].weights, num_samples_left) # Importance sampling by weight
			messages_next[t][s].pos[end_rand_ind:num_samples] = list_mvn(belief[s].pos[belief_inds], message_resample_cov, single_cov=True) # Add multivariate noise to the mean
			messages_next[t][s].orien[end_rand_ind:num_samples] = belief[s].orien[belief_inds] # Don't add any noise to the orientation (yet)
			messages_next[t][s].weights[end_rand_ind:num_samples] = 1.0 / num_samples

	# Weight messages based on their neighbors. If it's the first iteration, then no weighting is performed.
	if iter_num != 1:
		raw_weights = np.zeros((num_messages, num_samples))
		for i in range(0, num_messages):
			t = neighbor_pair_list[i][0]
			s = neighbor_pair_list[i][1]
			# Weight m_t->s
			raw_weights[i][:] = weightMessage(messages_next, messages_prev, t, s)
		# Normalize for each message (each row is for a message, so we sum along axis 1)
		raw_weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)
		# Assign weights to the samples in messages_next
		for i in range(0, num_messages):
			t = neighbor_pair_list[i][0]
			s = neighbor_pair_list[i][1]
			messages_next[t][s].weights = raw_weights[i][:]

	messages_prev = copy.deepcopy(messages_next)

	t1 = time.time()
	message_time = t1-t0
	write("Done! dt=%f\n" % message_time)

	#################
	# Belief Update #
	#################
	write("Performing belief update...")
	flush()
	t0 = time.time()

	for neighbor in neighbor_dict[0]:
		for i in range(0, num_samples):
			pos = messages_next[neighbor][0].pos[i]
			dist2 = np.sum(np.asarray(pos, dtype=float) ** 2)
			unary_weight = 1.0 / (1.0 + (10.0 * dist2))
			messages_next[neighbor][0].weights[i] = messages_next[neighbor][0].weights[i] * unary_weight
		messages_next[neighbor][0].weights = messages_next[neighbor][0].weights / np.sum(messages_next[neighbor][0].weights)

	for i in range(0, num_points):
		# First, update weights of every sample w_ts based on the unary potential
		# Because we don't have a unary potential, we don't actually do this step!
		pass
		# ACTUALLY this is being done a few lines up

		# Now, combine all incoming messages
		s = i
		neighbors = neighbor_dict[s][:]
		num_neighbors = len(neighbors)
		combined_message = Message(num_samples*num_neighbors, source_dim, target_dim)
		for j in range(0, num_neighbors):
			t = neighbors[j]
			start = j*num_samples
			stop = j*num_samples + num_samples
			combined_message.pos[start:stop] = messages_next[t][s].pos[:]
			combined_message.orien[start:stop] = messages_next[t][s].orien[:]
			combined_message.weights[start:stop] = messages_next[t][s].weights[:]
		combined_message.weights = combined_message.weights / sum(combined_message.weights) # Normalize

		# Resample from combined_message to get the belief
		message_inds = weightedSample(combined_message.weights, num_samples)
		belief[i].pos = combined_message.pos[message_inds]
		belief[i].orien = combined_message.orien[message_inds]
		belief[i].weights = combined_message.weights[message_inds]

	t1 = time.time()
	belief_time = t1-t0
	write("Done! dt=%f\n" % belief_time)

	################
	# Write Images #
	################
	write("Writing images...")
	flush()
	t0 = time.time()

	from visualization.plot_belief import plot_belief_1d, plot_mle_1d, plot_mean_1d
	from visualization.plot_message import plot_message_1d
	from matplotlib.collections import LineCollection
	from mpl_toolkits.mplot3d.art3d import Line3DCollection

	fig, ax = plt.subplots()
	plot_belief_1d(belief, 0, ax, show_mle=True, show_mean=True)
	plt.savefig(output_dir + ("bel0_iter%d.svg" % iter_num))
	plt.close(fig)

	fig, ax = plt.subplots()
	plot_belief_1d(belief, 1, ax, show_mle=True, show_mean=True)
	plt.savefig(output_dir + ("bel1_iter%d.svg" % iter_num))
	plt.close(fig)

	fig, ax = plt.subplots()
	plot_belief_1d(belief, 2, ax, show_mle=True, show_mean=True)
	plt.savefig(output_dir + ("bel2_iter%d.svg" % iter_num))
	plt.close(fig)

	fig, ax = plt.subplots()
	plot_message_1d(messages_next, 0, 1, ax)
	plt.savefig(output_dir + ("m0-1_iter%d.svg" % iter_num))
	plt.close(fig)

	fig, ax = plt.subplots()
	plot_message_1d(messages_next, 1, 0, ax)
	plt.savefig(output_dir + ("m1-0_iter%d.svg" % iter_num))
	plt.close(fig)

	fig, ax = plt.subplots()
	plot_message_1d(messages_next, 1, 2, ax)
	plt.savefig(output_dir + ("m1-2_iter%d.svg" % iter_num))
	plt.close(fig)

	fig, ax = plt.subplots()
	plot_message_1d(messages_next, 2, 1, ax)
	plt.savefig(output_dir + ("m2-1_iter%d.svg" % iter_num))
	plt.close(fig)

	point_size=3
	point_line_width=0.5
	line_width=0.25
	line_length=0.25
	line_color="black"

	fig, ax = plt.subplots()
	coordinates = np.zeros((num_points*num_samples, 2, 2))
	for i in range(num_points):
		for j in range(num_samples):
			c_idx = i*num_points + j
			coordinates[c_idx][0][0] = points[i][0]
			coordinates[c_idx][0][1] = points[i][1]
			coordinates[c_idx][1][0] = points[i][0] + (line_length * belief[i].orien[j][0][0])
			coordinates[c_idx][1][1] = points[i][1] + (line_length * belief[i].orien[j][0][1])
	ax.scatter(points[:,0], points[:,1], c=color, cmap=plt.cm.Spectral, s=point_size**2, zorder=2, linewidth=point_line_width)
	lines = LineCollection(coordinates, color=line_color, linewidths=line_width)
	ax.add_collection(lines)
	plt.savefig(output_dir + ("orien_iter%d.svg" % iter_num))
	plt.close(fig)

	t1 = time.time()
	image_time = t1-t0
	write("Done! dt=%f\n" % image_time)

	total_time = message_time + belief_time + image_time
	write("Total iteration time: %f\n" % total_time)

fig, ax = plt.subplots()
plot_neighbors_2d(points, color, neighbor_graph, ax)
from visualization.plot_reconstruction import plot_reconstruction_1d_2d
mlePoints = np.zeros(num_points)
for i in range(num_points):
	ind = np.argmax(belief[i].weights)
	# mlePoints[i] = belief[i].pos.flatten()[ind]
	mlePoints[i] = np.average(belief[i].pos.flatten(), weights=belief[i].weights)
plot_reconstruction_1d_2d(points, np.argsort(mlePoints), ax)
plt.show()

point_size=3
point_line_width=0.5
line_width=0.25
line_length=0.25
line_color="black"

from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

fig, ax = plt.subplots()
coordinates = np.zeros((num_points*num_samples, 2, 2))
for i in range(num_points):
	for j in range(num_samples):
		c_idx = i*num_points + j
		coordinates[c_idx][0][0] = points[i][0]
		coordinates[c_idx][0][1] = points[i][1]
		coordinates[c_idx][1][0] = points[i][0] + (line_length * belief[i].orien[j][0][0])
		coordinates[c_idx][1][1] = points[i][1] + (line_length * belief[i].orien[j][0][1])
ax.scatter(points[:,0], points[:,1], c=color, cmap=plt.cm.Spectral, s=point_size**2, zorder=2, linewidth=point_line_width)
lines = LineCollection(coordinates, color=line_color, linewidths=line_width)
ax.add_collection(lines)

plt.show()