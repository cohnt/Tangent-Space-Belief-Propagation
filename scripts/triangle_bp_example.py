import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy
from utils import write, flush
from sklearn.neighbors import kneighbors_graph

use_true_unary_potential = False
use_noisy_unary_potential = True
use_extra_noise = True
use_relative_location = True

num_iters = 20     # Number of iterations of the message passing algorithm to run
num_samples = 100  # Number of samples in each belief and message
explore_perc = 0

message_resample_cov = np.eye(2) * 0.1 # TODO: Change

output_dir = "other_results/results_triangle/"

write("\n")

actual_locations = np.array([[-1, -1], [0, 1.5], [1, -1], [1.5, 1.5], [-1.5, 0], [0, -0.5], [0, 2]])
neighbor_graph = kneighbors_graph(actual_locations, 2, mode="distance", n_jobs=-1)
neighbor_dict = {
	0: [1, 2],
	1: [0, 2],
	2: [0, 1]
}
neighbor_pair_list = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

num_nodes = 3
num_messages = 6

def displayPoints(points, weights, ax):
	ax.scatter(points[:,0], points[:,1], c=weights, cmap="jet")

def displayMessage(messages_mat, t, s, ax):
	displayPoints(messages_mat[t][s].pos, messages_mat[t][s].weights, ax)

#######################
# Initialize Messages #
#######################
from utils import Message

messages_prev = [[None for j in range(num_nodes)] for i in range(num_nodes)]
messages_next = [[None for j in range(num_nodes)] for i in range(num_nodes)]

for key, value in neighbor_pair_list:
	messages_prev[key][value] = Message(num_samples, 2)
	messages_prev[key][value].pos = np.random.uniform(-2, 2, (num_samples, 2))
	messages_prev[key][value].weights = np.zeros(num_samples) + (1.0 / float(num_samples))

	messages_next[key][value] = Message(num_samples, 2)

###################
# Message Passing #
###################
from utils import Belief, weightedSample, list_mvn, sphereRand

belief = [Belief(num_samples, 2) for _ in range(num_nodes)]

def weightFromNeighbor(m_next, m_prev, current, neighbor):
	weights_unary = np.zeros(num_samples)
	weights_prior = np.zeros(num_samples)
	weights = np.zeros(num_samples)

	s = current
	t = neighbor
	# m_t->s

	neighbors = neighbor_dict[t][:]
	neighbors.remove(s)
	num_neighbors = len(neighbors)

	for i in range(num_samples):
		pos_s = m_next[t][s].pos[i]
		pos_t_hat = sampleNeighbor(pos_s, s, t)

		if use_true_unary_potential:
			distError = np.linalg.norm(pos_t_hat - actual_locations[t])
			weights_unary[i] = 1.0 / (1.0 + distError)
		elif use_noisy_unary_potential:
			distErrors = []
			if use_extra_noise:
				upper_lim = len(actual_locations)
			else:
				upper_lim = num_nodes
			for obs_idx in range(upper_lim):
				distErrors.append(np.linalg.norm(pos_t_hat - actual_locations[obs_idx]))
			weights_unary[i] = 1.0 / (1.0 + np.min(distErrors))
		else:
			weights_unary[i] = 1.0

		if num_neighbors > 0:
			weights_from_priors = np.zeros(num_neighbors)
			for j in range(0, num_neighbors):
				u = neighbors[j]
				weights_from_priors[j] = weightFromPrior(pos_t_hat, m_prev, s, t, u)
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

def sampleNeighbor(pos, current, neighbor):
	s = current
	t = neighbor
	if use_relative_location:
		loc = pos - actual_locations[s] + actual_locations[t]
		return np.random.multivariate_normal(loc, np.eye(2) * 1.0)
	else:
		expectedDist = neighbor_graph[s,t]
		return sphereRand(pos, expectedDist)

def weightFromPrior(pos, m_prev, s, t, u):
	# Use m_u->t to help weight m_t->s
	weight_prior = 0
	for i in range(0, num_samples):
		pos_pred = m_prev[u][t].pos[i]
		dist2 = np.sum((np.asarray(pos, dtype=float) - np.asarray(pos_pred, dtype=float)) ** 2)
		weight_pairwise = 1.0/(1.0+dist2)
		weight_prior = weight_prior + (m_prev[u][t].weights[i] * weight_pairwise)
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
			if start_ind < end_rand_ind:
				messages_next[t][s].pos[start_ind:end_rand_ind] = np.random.uniform(-2, 2, (end_rand_ind - start_ind, 2))
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
			# m_t->s
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

	for i in range(0, num_nodes):
		# First, update weights of every sample w_ts based on the unary potential
		# Because we don't have a unary potential, we don't actually do this step!
		pass

		# Now, combine all incoming messages
		s = i
		neighbors = neighbor_dict[s][:]
		num_neighbors = len(neighbors)
		combined_message = Message(num_samples, 2)
		combined_message.pos = np.zeros((num_samples*num_neighbors, 2))
		combined_message.weights = np.zeros(num_samples*num_neighbors)
		for j in range(0, num_neighbors):
			t = neighbors[j]
			start = j*num_samples
			stop = j*num_samples + num_samples
			combined_message.pos[start:stop] = messages_next[t][s].pos[:]
			combined_message.weights[start:stop] = messages_next[t][s].weights[:]
		combined_message.weights = combined_message.weights / sum(combined_message.weights) # Normalize

		# Resample from combined_message to get the belief
		message_inds = weightedSample(combined_message.weights, num_samples)
		belief[i].pos = combined_message.pos[message_inds]
		belief[i].weights = combined_message.weights[message_inds]

	t1 = time.time()
	belief_time = t1-t0
	write("Done! dt=%f\n" % belief_time)

	# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
	# displayPoints(messages_next[0][1].pos, messages_next[0][1].weights, ax1)
	# displayPoints(messages_next[2][1].pos, messages_next[2][1].weights, ax2)
	# displayPoints(belief[1].pos, belief[1].weights, ax3)
	# plt.savefig(output_dir + ("node1_%d.svg" % iter_num))
	# plt.close(fig)

	# fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
	# displayPoints(belief[0].pos, belief[0].weights, ax0)
	# displayPoints(belief[1].pos, belief[1].weights, ax1)
	# displayPoints(belief[2].pos, belief[2].weights, ax2)
	# means = np.zeros((3, 2))
	# for i in range(3):
	# 	means[i,:] = np.average(belief[i].pos, axis=0, weights=belief[i].weights)
	# colors = [0, 0.5, 1]
	# displayPoints(means, colors, ax3)
	# plt.savefig(output_dir + ("beliefs_%d.svg" % iter_num))
	# plt.close(fig)

	# fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
	# displayPoints(belief[0].pos, "red", ax0)
	# displayPoints(belief[1].pos, "blue", ax0)
	# displayPoints(belief[2].pos, "green", ax0)
	# plt.show()

	fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
	displayPoints(belief[0].pos, belief[0].weights, ax0)
	displayPoints(belief[1].pos, belief[1].weights, ax1)
	displayPoints(belief[2].pos, belief[2].weights, ax2)
	displayPoints(belief[0].pos, "red", ax3)
	displayPoints(belief[1].pos, "blue", ax3)
	displayPoints(belief[2].pos, "green", ax3)
	plt.savefig(output_dir + ("beliefs_%d.svg" % iter_num))
	plt.close(fig)

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
