import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import time
import sys
import copy
from utils import write, flush

num_iters = 5      # Number of iterations of the message passing algorithm to run
neighbors_k = 6    # The value of 'k' used for k-nearest-neighbors
num_points = 100   # Number of data points
data_noise = 0.00001 # How much noise is added to the data
num_samples = 10   # Numbers of samples used in the belief propagation algorithm
explore_perc = 0.5 # Fraction of uniform samples to keep exploring
initial_dim = 2    # The dimensionality of the incoming dataset (see "Load Dataset" below)
target_dim = 1     # The number of dimensions the data is being reduced to

message_resample_cov = np.eye(target_dim) * 0.1 # TODO: Change

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
plot_neighbors_2d(points, color, neighbor_graph, ax, point_size=3, line_width=0.25, edge_thickness=0.25)
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
from utils import weightedSample, list_mvn, sphereRand

class Belief:
	def __init__(self):
		self.pos = np.zeros((num_samples, target_dim))
		self.weights = np.zeros(num_samples)

belief = [Belief() for _ in range(num_points)]

def weightFromNeighbor(m_next, m_prev, current, neighbor):
	weights_unary = np.zeros(num_samples)
	weights_prior = np.zeros(num_samples)
	weights = np.zeros(num_samples)

	neighbors = neighbor_dict[t][:]
	neighbors.remove(s)
	num_neighbors = len(neighbors)

	for i in range(0, num_samples):
		# We're dealing with m_t->s
		pos_s = m_next[t][s].pos[i]
		pos_t = sampleNeighbor(pos_s, s, t) # TODO: other params

		# TODO: Possibly check that pos_s[i] is valid (do we even have to?)
		# For now, assume pos_s[i] is indeed valid
		weights_unary[i] = 1.0 # For now, we don't actually have a unary potential
		                       # It's not even clear if we have an observation

		if num_neighbors > 0:
			# In theory, we don't need this check, since we're using k-nearest neighbors, so
			# every point is going to have at least k neighbors. But if, for some reason, a
			# node didn't have any neighbors, its pairwise potential would be the empty product.
			# And the empty product is defined to be 1.
			weights_from_priors = np.zeros(num_neighbors)
			for j in range(0, num_neighbors):
				u = neighbors[j]
				weights_from_priors[j] = weightFromPrior(pos_t, m_prev, u, t, s)
			weights_prior[i] = np.prod(weights_from_priors)

	# Normalize (if all zero, then make all weights equal)
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

	weights = np.multiply(weights_unary, weights_prior)
	return weights

def weightFromPrior(pos, m_prev, u, t, s):
	# Use m_u->t to help weight m_t->s
	weight_prior = 0
	for i in range(0, num_samples):
		pos_pred = m_prev[u][t].pos[i]
		dist2 = (np.asarray(pos, dtype=float) - np.asarray(pos_pred, dtype=float)) ** 2
		weight_pairwise = 1/(1+dist2)
		weight_prior = weight_prior + (m_prev[u][t].weights[i] * weight_pairwise)
	return weight_prior

def sampleNeighbor(pos, s, t):
	expectedDist = neighbor_graph[s,t]
	return sphereRand(pos, expectedDist)

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
		# Resample messages from previous belief
		for neighbor_pair in neighbor_pair_list:
			t = neighbor_pair[0]
			s = neighbor_pair[1]
			end_rand_ind = int(np.floor(num_samples * explore_perc))

			start_ind = 0
			max_weight_ind = np.argmax(belief[s].weights)
			max_weight = belief[s].weights[max_weight_ind]
			if max_weight != 1.0 / num_samples:
				# Not all samples have the same weight, so we keep the highest weighted sample
				start_ind = 1

			# Keep the best sample (if start_ind == 0, this will be an empty slice and nothing will happen)
			messages_next[t][s].pos[0:start_ind] = belief[s].pos[max_weight_ind][:]
			# messages_next[t][s].weights[0:start_ind] = belief[s].weights[max_weight_ind]
			messages_next[t][s].weights[0:start_ind] = 1.0 / num_samples

			# Sample [start_int, end_rand_in) uniformly across the state space, each with uniform weight
			messages_next[t][s].pos[start_ind:end_rand_ind] = np.random.uniform(-2, 2, (end_rand_ind - start_ind, 1))
			messages_next[t][s].weights[start_ind:end_rand_ind] = 1.0 / num_samples

			# Sample [end_rand_in, num_samples) based on the belief, with added noise
			num_samples_left = num_samples - end_rand_ind
			belief_inds = weightedSample(belief[s].weights, num_samples_left)
			messages_next[t][s].pos[end_rand_ind:num_samples] = list_mvn(belief[s].pos[belief_inds], message_resample_cov, single_cov=True)
			# messages_next[t][s].weights[end_rand_ind, num_samples] = belief[s].weights[belief_inds]
			messages_next[t][s].weights[end_rand_ind:num_samples] = 1.0 / num_samples

	if iter_num != 1:
		# Weight messages based on their neighbors.
		raw_weights = np.zeros((num_messages, num_samples))
		for i in range(0, num_messages):
			t = neighbor_pair_list[i][0]
			s = neighbor_pair_list[i][1]
			raw_weights[i][:] = weightFromNeighbor(messages_next, messages_prev, s, t)
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

write("\n")