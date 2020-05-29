import numpy as np
import scipy
import random

from textwrap import wrap
import matplotlib.pyplot as plt
import time
import sys
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import write, flush

global_t0 = time.time()

dataset_name = "long_spiral_curve"
dataset_seed = 4045775215 # np.random.randint(0, 2**32)
num_points = 5    # Number of data points
data_noise = 0 # How much noise is added to the data
source_dim = 2      # The dimensionality of the incoming dataset (see "Load Dataset" below)
target_dim = 1      # The number of dimensions the data is being reduced to
new_dim_list = [100, 110, 120, 130, 140, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000] # The higher dimension the data will be mapped to

num_iters = 10+1      # Number of iterations of the message passing algorithm to run for each dimension
neighbors_k = 3    # The value of 'k' used for k-nearest-neighbors
num_samples = 1     # Numbers of samples used in the belief propagation algorithm
explore_perc = 0.1  # Fraction of uniform samples to keep exploring

message_resample_cov = np.eye(target_dim) * 0.01 # TODO: Change
pruning_angle_thresh = np.cos(30.0 * np.pi / 180.0)
ts_noise_variance = 30 # In degrees

embedding_name = "KernelPCA" # Could also be MDS
kpca_eigen_solver = "auto"
kpca_tol = 1e-9
kpca_max_iter = 3000

time_by_dim = []

for new_dim in new_dim_list:
	iter_time_list = []
	print "\n\n\n"

	################
	# Load Dataset #
	################
	from datasets.dim_2.arc_curve import make_arc_curve
	from datasets.dim_2.s_curve import make_s_curve
	from datasets.dim_2.o_curve import make_o_curve
	from datasets.dim_2.eight_curve import make_eight_curve
	from datasets.dim_2.long_spiral_curve import make_long_spiral_curve

	write("Generating dataset...")
	flush()
	t0 = time.time()
	points, color, true_tangents, true_parameters, dataset_seed = make_long_spiral_curve(num_points, data_noise, rs_seed=dataset_seed)
	t1 = time.time()
	write("Done! dt=%f\n" % (t1-t0))
	flush()

	from utils import increaseDimensionMatrix
	write("Increasing dimenion.\n")
	flush()
	source_dim = len(points[0])
	mat = increaseDimensionMatrix(source_dim, new_dim)
	points = np.matmul(points, mat)
	true_tangents = np.matmul(true_tangents, mat)
	source_dim = new_dim

	#######################
	# k-Nearest-Neighbors #
	#######################
	from sklearn.neighbors import kneighbors_graph
	from visualization.plot_neighbors import plot_neighbors_3d
	from visualization.plot_pca import plot_pca_3d
	from visualization.animate import rotanimate

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

	write("Number of points: %d\n" % num_points)
	write("Number of edges: %d\n" % len(neighbor_pair_list))

	###############
	# Measure PCA #
	###############
	from sklearn.decomposition import PCA

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

	#######################
	# Initialize Messages #
	#######################
	from scipy.stats import special_ortho_group
	from utils import randomSmallRotation

	class Message():
		def __init__(self, num_samples, source_dim, target_dim):
			# If num_samples=N and source_dim=n, and target_dim=m, then:
			# self.ts is a list of ordered bases of m-dimensional (i.e. spanned by m
			# unit vectors) subspaces in R^n, so it's of shape (N, m, n)
			# self.weights is a list of weights, so it's of shape (N)
			self.ts = np.zeros((num_samples, target_dim, source_dim))
			self.weights = np.zeros(num_samples)

	def randomTangentSpaceList(num_samples, source_dim, target_dim):
		# Return a random list of size target_dim orthonormal vectors in source_dim.
		# This represents the basis of a random subspace of dimension target_dim in
		# the higher dimensional space of dimension source_dim
		ts = np.zeros((num_samples, target_dim, source_dim))
		for i in range(num_samples):
			ts[i][:] = special_ortho_group.rvs(dim=source_dim)[0:target_dim]
		return ts

	def noisifyTS(ts, var):
		rotMat = randomSmallRotation(source_dim, variance=var)
		# theta = np.random.normal(0, var) * np.pi / 180.0
		# rotMat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
		return np.array([np.dot(rotMat, ts[0])])

	def noisifyTSList(ts_list, var=5):
		for i in range(len(ts_list)):
			ts_list[i] = noisifyTS(ts_list[i], var)
		return ts_list

	# This initializes messages_prev and messages_next as num_points by num_points arrays of Nones.
	# Where appropriate, the Nones will be replaced by Message objects
	messages_prev = [[None for __ in range(num_points)] for _ in range(num_points)]
	messages_next = [[None for __ in range(num_points)] for _ in range(num_points)]
	for key, value in neighbor_pair_list:
		# Note that key represents where the message is coming from and value represents where the message is going to
		# In other words, messages[key][value] === m_key->value
		messages_prev[key][value] = Message(num_samples, source_dim, target_dim)
		# messages_prev[key][value].ts = randomTangentSpaceList(num_samples, source_dim, target_dim)
		messages_prev[key][value].ts = noisifyTSList(np.repeat([observations[value]], num_samples, axis=0), var=ts_noise_variance)
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
		# neighbors.remove(s)
		num_neighbors = len(neighbors)

		for i in range(num_samples):
			ts_s = m_next[t][s].ts[i]
			ts_t = sampleNeighbor(ts_s, t, s)

			weights_unary[i] = weightUnary(ts_t, t)

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
		# principal_angles = subspace_angles(ts.transpose(), observations[idx].transpose())
		# total_angle_error = np.sum(principal_angles)
		# weight = 1.0 / (1.0 + total_angle_error)

		# dprod = 1 - np.abs(np.dot(ts[0], observations[idx][0]))
		# weight = 1.0 / (1.0 + dprod)

		dist = compareSubspaces(ts, observations[idx])
		weight = 1.0 / (1.0 + dist)

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
			# principal_angles = subspace_angles(ts_s.transpose(), ts_t.transpose())
			# weight = 1.0 / (1.0 + np.sum(principal_angles))
			
			# dprod = 1 - np.abs(np.dot(ts_s[0], ts_t[0]))
			# weight = 1.0 / (1.0 + dprod)

			dist = compareSubspaces(ts_s, ts_t)
			weight = 1.0 / (1.0 + dist)

			weight_prior = weight_prior + (m_prev[u][t].weights[i] * weight)
		return weight_prior

	def compareSubspaces(basis1, basis2):
		total = 0
		for vec in basis1:
			p_vec = np.dot(projSubspace(basis2, vec), basis2)
			diff = vec - p_vec
			total = total + np.dot(diff, diff)
		return total

	def resampleMessage(t, s):
		start_ind = 0
		max_weight_ind = np.argmax(belief[s].weights)
		max_weight = belief[s].weights[max_weight_ind]
		# if max_weight != 1.0 / num_samples:
		# 	# Not all samples have the same weight, so we keep the highest weighted sample
		# 	start_ind = 1

		# Note that we don't actually care about the weights we are assigning in this step, since
		# all the samples will be properly weighted later on. In theory, we don't have to assign
		# the samples weights at all, but it seems natural to at least give them the "default"
		# weight of 1.0 / num_samples.

		# If there is a best sample, keep it. In theory, this could be expanded to take the best
		# n samples, with only a little modification. One could use argsort to sort the array, but
		# this would be a pretty big performance hit. This stackoverflow answer suggests using
		# argpartition instead: https://stackoverflow.com/a/23734295/
		if start_ind == 1:
			messages_next[t][s].ts[0:start_ind] = belief[s].ts[max_weight_ind][:]
			messages_next[t][s].weights[0:start_ind] = 1.0 / num_samples

		# Some samples will be randomly seeded across the state space. Specifically, the
		# interval [start_ind, end_rand_in). This will be slightly less than explore_perc
		# if a maximum weight value is kept from the previous iteration.
		end_rand_ind = int(np.floor(num_samples * explore_perc))
		this_section_num_samples = end_rand_ind - start_ind
		# If explore_perc is so small (or just zero) that the number of random samples
		# is 0, then we don't need to do this step.
		if this_section_num_samples > 0:
			messages_next[t][s].ts[start_ind:end_rand_ind] = randomTangentSpaceList(this_section_num_samples, source_dim, target_dim)
			messages_next[t][s].weights[start_ind:end_rand_ind] = 1.0 / num_samples

		# Finally, we generate the remaining samples (i.e. the interval [end_rand_in, num_samples))
		# by resampling from the belief of the previous iteration, with a little added noise.
		num_samples_left = num_samples - end_rand_ind
		belief_inds = weightedSample(belief[s].weights, num_samples_left) # Importance sampling by weight
		messages_next[t][s].ts[end_rand_ind:num_samples] = noisifyTSList(belief[s].ts[belief_inds]) # Don't add any noise to the orientation (yet)
		messages_next[t][s].weights[end_rand_ind:num_samples] = 1.0 / num_samples

	def evalError(true_tangents, estimated_tangents):
		error_arr = np.zeros(len(true_tangents))
		for i in range(len(true_tangents)):
			error_arr[i] = compareSubspaces(true_tangents[i], estimated_tangents[i])
		max_error = np.max(error_arr)
		mean_error = np.mean(error_arr)
		median_error = np.median(error_arr)
		return (max_error, mean_error, median_error, error_arr)

	parallel = Parallel(n_jobs=-1, verbose=0, backend="threading")
	max_errors = []
	mean_errors = []
	median_errors = []

	for iter_num in range(1, num_iters+1):
		write("\nIteration %d\n" % iter_num)

		message_time = 0
		belief_time = 0
		image_time = 0
		graph_time = 0
		total_time = 0

		##################
		# Message Update #
		##################
		write("Performing message update...\n")
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
				resampleMessage(t, s)

		# Weight messages based on their neighbors. If it's the first iteration, then no weighting is performed.
		if iter_num != 1:
			# raw_weights = np.zeros((num_messages, num_samples))
			# for i in range(0, num_messages):
			# 	t = neighbor_pair_list[i][0]
			# 	s = neighbor_pair_list[i][1]
			# 	# Weight m_t->s
			# 	raw_weights[i][:] = weightMessage(messages_next, messages_prev, t, s)

			raw_weights = np.asarray(parallel(delayed(weightMessage)(messages_next, messages_prev, neighbor_pair_list[i][0], neighbor_pair_list[i][1]) for i in tqdm(range(num_messages))))

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
				combined_message.ts[start:stop] = messages_next[t][s].ts[:]
				combined_message.weights[start:stop] = messages_next[t][s].weights[:]
			combined_message.weights = combined_message.weights / sum(combined_message.weights) # Normalize

			# Resample from combined_message to get the belief
			message_inds = weightedSample(combined_message.weights, num_samples)
			belief[i].ts = combined_message.ts[message_inds]
			belief[i].weights = combined_message.weights[message_inds]

		t1 = time.time()
		belief_time = t1-t0
		write("Done! dt=%f\n" % belief_time)

		total_time = message_time + belief_time
		write("Total iteration time: %f\n" % total_time)
		if iter_num > 1:
			iter_time_list.append(total_time)

	print iter_time_list
	print np.average(iter_time_list)
	time_by_dim.append(np.average(iter_time_list))
	print "Average iteration time for dimension %d: %f" % (new_dim, time_by_dim[-1])

print time_by_dim

fig = plt.figure()
plt.plot(new_dim_list, time_by_dim)
plt.title("Time by Dimension")
plt.savefig("time_by_dimension.png")
plt.close(fig)

fig = plt.figure()
plt.plot(np.log(new_dim_list), np.log(time_by_dim))
plt.title("Time by Dimension (Log-Log)")
plt.savefig("time_by_dimension_log_log.png")
plt.close(fig)

from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(new_dim_list), np.log(time_by_dim))
print "Slope: %f" % slope
print "r: %f" % r_value