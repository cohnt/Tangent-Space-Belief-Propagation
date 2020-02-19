import numpy as np
import scipy
import random
import matplotlib
matplotlib.use('Agg')
from textwrap import wrap
import matplotlib.pyplot as plt
import time
import sys
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import write, flush, pairwiseDistErr
from collections import OrderedDict

global_t0 = time.time()

dataset_name = "swiss_roll"
dataset_seed = np.random.randint(0, 2**32)
num_points = 350    # Number of data points
data_noise = 0.001     # How much noise is added to the data
source_dim = 3      # The dimensionality of the incoming dataset (see "Load Dataset" below)
target_dim = 2      # The number of dimensions the data is being reduced to

num_iters = 25      # Number of iterations of the message passing algorithm to run
neighbors_k = 6    # The value of 'k' used for k-nearest-neighbors
num_samples = 10     # Numbers of samples used in the belief propagation algorithm
explore_perc = 0.1  # Fraction of uniform samples to keep exploring

message_resample_cov = np.eye(target_dim) * 0.01 # TODO: Change
pruning_angle_thresh = np.cos(30.0 * np.pi / 180.0)
ts_noise_variance = 30 # Degrees
initial_variance = 30 # Degree

output_dir = "results_sheet/"
error_histogram_num_bins = num_points / 10
err_dist_metric = "l2"
err_mat_norm = "max"

embedding_name = "KernelPCA" # Could also be MDS
kpca_eigen_solver = "auto"
kpca_tol = 1e-9
kpca_max_iter = 3000

data_sp_rad = 5.0
data_sp_lw = 1.0
nn_lw = 0.5
pca_ll = 0.1
embedding_sp_rad = 7.0
embedding_sp_lw = 1.0
combined_sp_rad = 4.0
combined_sp_lw = 0.5
disp_elev = 5.0
disp_azim = -85.0

write("\n")

def make3DFigure():
	f = plt.figure(figsize=(14.4, 10.8), dpi=100)
	a = f.add_subplot(111, projection='3d')
	if dataset_name == "swiss_roll":
		a.set_ylim(bottom=-0.5, top=1.5)
	a.view_init(elev=disp_elev, azim=disp_azim)
	return f, a

matplotlib.rcParams.update({'font.size': 15})

####################
# Write Parameters #
####################
f = open(output_dir + "parameters.ini", "w")

# Write as an INI file, so it can be directly entered into another program later.
# [Section Name]
# ; Comment
# Key = Value

f.write("[Dataset]\n")
f.write("name=%s\n" % dataset_name)
f.write("seed=%d\n" % dataset_seed)
f.write("num_points=%d\n" % num_points)
f.write("noise=%s\n" % str(data_noise))
f.write("source_dim=%d\n" % source_dim)
f.write("target_dim=%d\n" % target_dim)

f.write("\n[Belief Propagation]\n")
f.write("max_iters=%d\n" % num_iters)
f.write("num_neighbors=%d\n" % neighbors_k)
f.write("num_samples=%d\n" % num_samples)
f.write("explore=%s\n" % str(explore_perc))
f.write("prune_thresh=%s\n" % str(pruning_angle_thresh))
f.write("ts_noise_variance=%s\n" % str(ts_noise_variance))
f.write("initial_variance=%s\n" % str(initial_variance))

f.write("\n[Embedding]\n")
f.write("embedding_method=%s\n" % embedding_name)
f.write("embedding_eigen_solver=%s\n" % kpca_eigen_solver)
f.write("embedding_tol=%s\n" % str(kpca_tol))
f.write("embedding_max_iter=%d\n" % kpca_max_iter)

f.write("\n[Display]\n")
f.write("data_sp_rad=%s\n" % str(data_sp_rad))
f.write("data_sp_lw=%s\n" % str(data_sp_lw))
f.write("nn_lw=%s\n" % str(nn_lw))
f.write("pca_ll=%s\n" % str(pca_ll))
f.write("embedding_sp_rad=%s\n" % str(embedding_sp_rad))
f.write("embedding_sp_lw=%s\n" % str(embedding_sp_lw))
f.write("combined_sp_rad=%s\n" % str(combined_sp_rad))
f.write("combined_sp_lw=%s\n" % str(combined_sp_lw))
f.write("disp_elev=%s\n" % str(disp_elev))
f.write("disp_azim=%s\n" % str(disp_azim))

f.close()

################
# Load Dataset #
################
from datasets.dim_3.s_sheet import make_s_sheet
from datasets.dim_3.swiss_roll import make_swiss_roll_sheet
from mpl_toolkits.mplot3d import Axes3D

write("Generating dataset...")
flush()
t0 = time.time()
points, color, true_tangents, true_parameters, dataset_seed = make_swiss_roll_sheet(num_points, data_noise, rs_seed=dataset_seed)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

write("Saving dataset plot...")
flush()
t0 = time.time()
fig, ax = make3DFigure()
ax.scatter(points[:,0], points[:,1], points[:,2], c=color, cmap=plt.cm.Spectral, s=data_sp_rad**2, linewidth=data_sp_lw)
ax.set_title("Dataset (num=%d, variance=%f, seed=%d)" % (num_points, data_noise, dataset_seed))
plt.savefig(output_dir + "dataset.svg")
plt.close(fig)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

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

write("Saving nearest neighbors plot...")
flush()
t0 = time.time()
fig, ax = make3DFigure()
# Not squared because it's squared inside plot_neighbors_3d
plot_neighbors_3d(points, color, neighbor_graph, ax, point_size=data_sp_rad, line_width=data_sp_lw, edge_thickness=nn_lw, show_labels=False, line_color="grey")
ax.set_title("Nearest Neighbors (k=%d)" % neighbors_k)
plt.savefig(output_dir + "nearest_neighbors.svg")
angles = np.linspace(0, 360, 40+1)[:-1]
rotanimate(ax, angles, output_dir + "nearest_neighbors.gif", delay=30, width=14.4, height=10.8, folder=output_dir, elevation=disp_elev)
plt.close(fig)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

write("Saving ground truth tangent plot...")
flush()
t0 = time.time()
fig, ax = make3DFigure()
# Not squared because it's squared inside plot_neighbors_3d
plot_pca_3d(points, color, true_tangents, ax, point_size=data_sp_rad, point_line_width=data_sp_lw, line_width=nn_lw, line_length=pca_ll)
ax.set_title("Exact Tangents")
plt.savefig(output_dir + "true_tangents.svg")
plt.close(fig)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

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

write("Saving PCA observations plot...")
flush()
t0 = time.time()
fig, ax = make3DFigure()
# Not squared because it's squared inside plot_neighbors_3d
plot_pca_3d(points, color, observations, ax, point_size=data_sp_rad, point_line_width=data_sp_lw, line_width=nn_lw, line_length=pca_ll)
ax.set_title("Measured Tangent Spaces (PCA)")
plt.savefig(output_dir + "pca_observations.svg")
plt.close(fig)
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
	return np.array([np.dot(rotMat, ts[0]), np.dot(rotMat, ts[1])])

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
	messages_prev[key][value].ts = noisifyTSList(np.repeat([observations[value]], num_samples, axis=0), var=initial_variance)
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

try:
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

		################
		# Write Images #
		################
		from mpl_toolkits.mplot3d.art3d import Line3DCollection
		from matplotlib.cm import coolwarm
		write("Writing images...")
		flush()
		t0 = time.time()

		mle_bases = np.zeros((num_points, target_dim, source_dim))
		for i in range(num_points):
			max_ind = np.argmax(belief[i].weights)
			mle_bases[i] = belief[i].ts[max_ind]

		fig, ax = make3DFigure()
		ax.view_init(elev=0, azim=-90)
		# Not squared because it's squared inside plot_neighbors_3d
		plot_pca_3d(points, color, mle_bases, ax, point_size=data_sp_rad, point_line_width=data_sp_lw, line_width=nn_lw, line_length=pca_ll)
		ax.set_title("Tangent Space MLE (iter %d)" % iter_num)
		plt.savefig(output_dir + ("ts_mle_iter%s.svg" % str(iter_num).zfill(4)))
		plt.close(fig)

		fig, ax = make3DFigure()
		ax.view_init(elev=0, azim=-90)
		ax.scatter(points[:,0], points[:,1], points[:,2], c=color, cmap=plt.cm.Spectral, s=data_sp_rad**2, linewidth=data_sp_lw)

		num_comps = len(belief[0].ts[0])
		coordinates = np.zeros((num_points*num_samples*num_comps, 2, source_dim))
		colors = np.zeros((num_points*num_samples*num_comps, 4))
		for i in range(num_points):
			max_weight = np.max(belief[i].weights)
			for j in range(num_samples):
				for k in range(num_comps):
					c_idx = i*num_samples*num_comps + j*num_comps + k
					coordinates[c_idx][0][0] = points[i][0]
					coordinates[c_idx][0][1] = points[i][1]
					coordinates[c_idx][0][2] = points[i][2]
					coordinates[c_idx][1][0] = points[i][0] + (pca_ll * belief[i].ts[j][k][0])
					coordinates[c_idx][1][1] = points[i][1] + (pca_ll * belief[i].ts[j][k][1])
					coordinates[c_idx][1][2] = points[i][2] + (pca_ll * belief[i].ts[j][k][2])
					colors[c_idx][:] = coolwarm(belief[i].weights[j] * (1.0 / max_weight))
		lines = Line3DCollection(coordinates, color=colors, linewidths=nn_lw)
		ax.add_collection(lines)
		ax.set_title("Tangent Space Belief (iter %d)" % iter_num)
		plt.savefig(output_dir + ("ts_bel_iter%s.svg" % str(iter_num).zfill(4)))
		plt.close(fig)

		t1 = time.time()
		image_time = t1-t0
		write("Done! dt=%f\n" % image_time)

		max_error, mean_error, median_error, error_data = evalError(true_tangents, mle_bases)
		max_errors.append(max_error)
		mean_errors.append(mean_error)
		median_errors.append(median_error)

		write("Rewriting graphs...")
		flush()
		t0 = time.time()

		# max_errors = np.array(max_errors)
		# mean_errors = np.array(mean_errors)
		# median_errors = np.array(median_errors)

		raw_max_error, raw_mean_error, raw_median_error, raw_error_data = evalError(true_tangents, observations)
		iters_array = np.arange(1, len(max_errors)+1)

		# Max error plot
		fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
		ax.plot(iters_array, max_errors)
		ax.axhline(y=raw_max_error, linewidth=3, color="red", linestyle="--")
		label_text = "Only PCA Error=%f" % raw_max_error
		ax.text(0.05, raw_max_error+(0.05 * max_errors[0]), label_text)
		ax.set_xlim(left=0)
		ax.set_ylim(bottom=0)
		ax.set_title("Maximum Tangent Space Error by Iteration")
		plt.xlabel("Iteration Number")
		plt.ylabel("Maximum Tangent Space Error")
		plt.savefig(output_dir + "max_error.svg")
		plt.close(fig)

		# Mean error plot
		fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
		ax.plot(iters_array, mean_errors)
		ax.axhline(y=raw_mean_error, linewidth=3, color="red", linestyle="--")
		label_text = "Only PCA Error=%f" % raw_mean_error
		ax.text(0.05, raw_mean_error+(0.05 * mean_errors[0]), label_text)
		ax.set_xlim(left=0)
		ax.set_ylim(bottom=0)
		ax.set_title("Mean Tangent Space Error by Iteration")
		plt.xlabel("Iteration Number")
		plt.ylabel("Mean Tangent Space Error")
		plt.savefig(output_dir + "mean_error.svg")
		plt.close(fig)

		# Median error plot
		fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
		ax.plot(iters_array, median_errors)
		ax.axhline(y=raw_median_error, linewidth=3, color="red", linestyle="--")
		label_text = "Only PCA Error=%f" % raw_median_error
		ax.text(0.05, raw_median_error+(0.05 * median_errors[0]), label_text)
		ax.set_xlim(left=0)
		ax.set_ylim(bottom=0)
		ax.set_title("Median Tangent Space Error by Iteration")
		plt.xlabel("Iteration Number")
		plt.ylabel("Median Tangent Space Error")
		plt.savefig(output_dir + "median_error.svg")
		plt.close(fig)

		# Iteration error histogram
		fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
		ax.hist(error_data, np.arange(0, 1, 1.0/error_histogram_num_bins))
		ax.set_title("Histogram of Tangent Space Error (iter %d)" % iter_num)
		ax.set_xlim(left=0, right=1)
		ax.set_ylim(top=num_points)
		plt.xlabel("Tangent Space Error")
		plt.ylabel("Count")
		plt.savefig(output_dir + ("error_histogram%s.svg" % str(iter_num).zfill(4)))

		t1 = time.time()
		graph_time = t1-t0
		write("Done! dt=%f\n" % graph_time)
		flush()

		total_time = message_time + belief_time + image_time + graph_time
		write("Total iteration time: %f\n" % total_time)
except KeyboardInterrupt:
	write("\nTerminating early after %d iterations.\n" % (iter_num-1))
	write("Iteration %d not completed.\n\n" % iter_num)
	flush()



write("Saving PCA error histogram...")
flush()
t0 = time.time()
raw_max_error, raw_mean_error, raw_median_error, raw_error_data = evalError(true_tangents, observations)
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.hist(raw_error_data, np.arange(0, 1, 1.0/error_histogram_num_bins))
ax.set_title("Histogram of PCA Error")
ax.set_xlim(left=0, right=1)
ax.set_ylim(top=num_points)
plt.xlabel("PCA Error")
plt.ylabel("Count")
plt.savefig(output_dir + "pca_error_histogram.svg")
plt.close(fig)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

write("Pruning edges...")
flush()
t0 = time.time()

pruned_neighbors = scipy.sparse.lil_matrix(neighbor_graph)
for i in range(0, num_points):
	for j in range(0, num_points):
		if i != j:
			vec = points[j] - points[i]
			proj_vec = projSubspace(mle_bases[i], vec)
			dprod = np.abs(np.dot(vec, np.dot(proj_vec, mle_bases[i])) / (np.linalg.norm(vec) * np.linalg.norm(proj_vec)))
			if dprod < pruning_angle_thresh:
				pruned_neighbors[i,j] = 0
				pruned_neighbors[j,i] = 0

t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

fig, ax = make3DFigure()
plot_neighbors_3d(points, color, pruned_neighbors, ax, point_size=data_sp_rad, line_width=data_sp_lw, edge_thickness=nn_lw, show_labels=False, line_color="grey")
ax.set_title("Pruned Nearest Neighbors (k=%d, thresh=%f)" % (neighbors_k, pruning_angle_thresh))
plt.savefig(output_dir + "pruned_nearest_neighbors.svg")
angles = np.linspace(0, 360, 40+1)[:-1]
rotanimate(ax, angles, output_dir + "pruned_nearest_neighbors.gif", delay=30, width=14.4, height=10.8, folder=output_dir, elevation=disp_elev)
plt.close(fig)

# Uses the disjoint-set datatype
# http://p-nand-q.com/python/data-types/general/disjoint-sets.html
class DisjointSet():
	def __init__(self, vals):
		self.sets = set([self.makeSet(elt) for elt in vals])
	def makeSet(self, elt):
		return frozenset([elt])
	def findSet(self, elt):
		for subset in self.sets:
			if elt in subset:
				return subset
	def union(self, set_a, set_b):
		self.sets.add(frozenset.union(set_a, set_b))
		self.sets.remove(set_a)
		self.sets.remove(set_b)
	def write(self):
		for subset in self.sets:
			print subset

connected_components = DisjointSet(range(num_points))
for i in range(num_points):
	for j in range(num_points):
		if pruned_neighbors[i,j] != 0:
			set_i = connected_components.findSet(i)
			if not j in set_i:
				set_j = connected_components.findSet(j)
				connected_components.union(set_i, set_j)

if len(connected_components.sets) == 1:
	write("Graph already connected!\n")
	flush()
else:
	write("There are %d connected components.\n" % len(connected_components.sets))
	while len(connected_components.sets) > 1:
		min_edge_idx = (-1, -1)
		min_edge_length = np.Inf
		for set_a in connected_components.sets:
			for set_b in connected_components.sets:
				if set_a != set_b:
					for i in set_a:
						for j in set_b:
							dist = np.linalg.norm(points[i] - points[j])
							if dist < min_edge_length:
								min_edge_length = dist
								min_edge_idx = (i, j)
		i = min_edge_idx[0]
		j = min_edge_idx[1]
		pruned_neighbors[i, j] = min_edge_length
		pruned_neighbors[j, i] = min_edge_length
		set_a = connected_components.findSet(i)
		set_b = connected_components.findSet(j)
		connected_components.union(set_a, set_b)

t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

fig, ax = make3DFigure()
plot_neighbors_3d(points, color, pruned_neighbors, ax, point_size=data_sp_rad, line_width=data_sp_lw, edge_thickness=nn_lw, show_labels=False, line_color="grey")
ax.set_title("Added Edges after Pruning")
plt.savefig(output_dir + "added_edges.svg")
angles = np.linspace(0, 360, 40+1)[:-1]
rotanimate(ax, angles, output_dir + "added_edges.gif", delay=30, width=14.4, height=10.8, folder=output_dir, elevation=disp_elev)
plt.close(fig)

from sklearn.utils.graph_shortest_path import graph_shortest_path
write("Finding shortest paths...")
flush()
t0 = time.time()
shortest_distances = graph_shortest_path(pruned_neighbors, directed=False)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

write("Fitting KernelPCA...")
flush()
t0 = time.time()

# from sklearn.manifold import MDS
# mds = MDS(n_components=target_dim, max_iter=3000, eps=1e-9, n_init=25, dissimilarity="precomputed", n_jobs=-1, metric=True)
# feature_coords = mds.fit_transform(shortest_distances)

from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=target_dim, kernel="precomputed", eigen_solver=kpca_eigen_solver, tol=kpca_tol, max_iter=kpca_max_iter, n_jobs=-1)
feature_coords = kpca.fit_transform((shortest_distances**2) * -0.5)

t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

tsbp_err = pairwiseDistErr(feature_coords, true_parameters, dist_metric=err_dist_metric, mat_norm=err_mat_norm)
print "TSBP Error: %f" % tsbp_err

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], feature_coords[:,1], c=color, cmap=plt.cm.Spectral, s=embedding_sp_rad**2, linewidths=embedding_sp_lw)
ax.set_title("2D Embedding from BP Tangent Correction")
plt.savefig(output_dir + "coord_bp.svg")
plt.close(fig)

from visualization.error_plots import regressionErrorCharacteristic, listRegressionErrorCharacteristic

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
regressionErrorCharacteristic(ax, feature_coords, true_parameters, dist_metric=err_dist_metric)
ax.set_title("\n".join(wrap("Regression Error Characteristic from BP Tangent Correction for Edge Pruning", 50)))
plt.savefig(output_dir + "rec_coord_bp.svg")
plt.close(fig)

############################
# Compare to Other Methods #
############################

method_errs = OrderedDict()
method_errs["TSBP"] = tsbp_err

embeddings_list = []
embeddings_name_list = []
embeddings_list.append(feature_coords)
embeddings_name_list.append("TSBP")

write("\nComparing to other methods...\n")
flush()

from sklearn.manifold import LocallyLinearEmbedding, MDS, Isomap, SpectralEmbedding, TSNE
from ltsa import compute_ltsa

methods = []
methods.append(LocallyLinearEmbedding(n_neighbors=neighbors_k, n_components=target_dim, n_jobs=-1))
methods.append(MDS(n_components=target_dim, n_jobs=-1))
methods.append(Isomap(n_neighbors=neighbors_k, n_components=target_dim, n_jobs=-1))
methods.append(SpectralEmbedding(n_components=target_dim, n_neighbors=neighbors_k, n_jobs=-1))
methods.append(TSNE(n_components=target_dim))
num_methods = len(methods)

method_names = ["LLE", "MDS", "Isomap", "SpectralEmbedding", "t-SNE"]

for i in range(num_methods):
	solver = methods[i]
	name = method_names[i]
	write("Computing %s..." % name)
	flush()
	t0 = time.time()
	feature_coords = solver.fit_transform(points)
	t1 = time.time()
	write("Done! dt=%f\n" % (t1-t0))
	flush()

	method_errs[name] = pairwiseDistErr(feature_coords, true_parameters, dist_metric=err_dist_metric, mat_norm=err_mat_norm)
	print "%s Error: %f" % (name, method_errs[name])

	embeddings_list.append(feature_coords)
	embeddings_name_list.append(name)
	
	fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
	ax.scatter(feature_coords[:,0], feature_coords[:,1], c=color, cmap=plt.cm.Spectral, s=embedding_sp_rad**2, linewidths=embedding_sp_lw)
	ax.set_title("\n".join(wrap("2D Embedding from %s" % name, 60)))
	plt.savefig(output_dir + ("comparison_%s.svg" % name))
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
	regressionErrorCharacteristic(ax, feature_coords, true_parameters, dist_metric=err_dist_metric)
	ax.set_title("\n".join(wrap("Regression Error Characteristic from %s" % name, 50)))
	plt.savefig(output_dir + ("rec_%s.svg" % name))
	plt.close(fig)

write("Computing Classical LTSA...")
flush()
t0 = time.time()
feature_coords = compute_ltsa(points, neighbor_dict, observations, source_dim, target_dim)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

method_errs["LTSA"] = pairwiseDistErr(feature_coords, true_parameters, dist_metric=err_dist_metric, mat_norm=err_mat_norm)
print "LTSA Error: %f" % method_errs["LTSA"]

embeddings_list.append(feature_coords)
embeddings_name_list.append("LTSA")

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], feature_coords[:,1], c=color, cmap=plt.cm.Spectral, s=embedding_sp_rad**2, linewidths=embedding_sp_lw)
ax.set_title("\n".join(wrap("2D Embedding from Classical LTSA", 60)))
plt.savefig(output_dir + "comparison_orig_LTSA.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
regressionErrorCharacteristic(ax, feature_coords, true_parameters, dist_metric=err_dist_metric)
ax.set_title("\n".join(wrap("Regression Error Characteristic from Classical LTSA", 50)))
plt.savefig(output_dir + "rec_orig_LTSA.svg")
plt.close(fig)

write("Computing LTSA with Tangent Space Correction...")
flush()
t0 = time.time()
feature_coords = compute_ltsa(points, neighbor_dict, mle_bases, source_dim, target_dim)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

method_errs["LTSA BPT"] = pairwiseDistErr(feature_coords, true_parameters, dist_metric=err_dist_metric, mat_norm=err_mat_norm)
print "LTSA BPT Error: %f" % method_errs["LTSA BPT"]

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], feature_coords[:,1], c=color, cmap=plt.cm.Spectral, s=embedding_sp_rad**2, linewidths=embedding_sp_lw)
ax.set_title("\n".join(wrap("2D Embedding from LTSA with Tangent Space Correction", 60)))
plt.savefig(output_dir + "comparison_corrected_LTSA.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
regressionErrorCharacteristic(ax, feature_coords, true_parameters, dist_metric=err_dist_metric)
ax.set_title("\n".join(wrap("Regression Error Characteristic from LTSA with Tangent Space Correction", 50)))
plt.savefig(output_dir + "rec_corrected_LTSA.svg")
plt.close(fig)

write("Computing LTSA with Tangent Space Correction and Edge Pruning...")
flush()
t0 = time.time()
pruned_neighbor_dict = sparseMatrixToDict(pruned_neighbors)
feature_coords = compute_ltsa(points, pruned_neighbor_dict, mle_bases, source_dim, target_dim)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

method_errs["LTSA Pruning"] = pairwiseDistErr(feature_coords, true_parameters, dist_metric=err_dist_metric, mat_norm=err_mat_norm)
print "LTSA Pruning Error: %f" % method_errs["LTSA Pruning"]

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], feature_coords[:,1], c=color, cmap=plt.cm.Spectral, s=embedding_sp_rad**2, linewidths=embedding_sp_lw)
ax.set_title("\n".join(wrap("2D Embedding from LTSA with Tangent Space Correction and Edge Pruning", 60)))
plt.savefig(output_dir + "comparison_pruned_LTSA.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
regressionErrorCharacteristic(ax, feature_coords, true_parameters, dist_metric=err_dist_metric)
ax.set_title("\n".join(wrap("Regression Error Characteristic from LTSA with Tangent Space Correction and Edge Pruning", 50)))
plt.savefig(output_dir + "rec_pruned_LTSA.svg")
plt.close(fig)

write("Computing HLLE...")
flush()
t0 = time.time()
feature_coords = LocallyLinearEmbedding(n_neighbors=neighbors_k, n_components=target_dim, n_jobs=-1, method="hessian", eigen_solver="dense").fit_transform(points)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

method_errs["HLLE"] = pairwiseDistErr(feature_coords, true_parameters, dist_metric=err_dist_metric, mat_norm=err_mat_norm)
print "HLLE Error: %f" % method_errs["HLLE"]

embeddings_list.append(feature_coords)
embeddings_name_list.append("HLLE")

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], feature_coords[:,1], c=color, cmap=plt.cm.Spectral, s=embedding_sp_rad**2, linewidths=embedding_sp_lw)
ax.set_title("\n".join(wrap("Actual Parameter Value vs Embedded Coordinate from HLLE\n Reconstruction Error: %f" % method_errs["HLLE"], 50)))
plt.xlabel("Actual Parameter Value")
plt.ylabel("Embedded Coordinate")
plt.savefig(output_dir + "comparison_HLLE.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
regressionErrorCharacteristic(ax, feature_coords, true_parameters, dist_metric=err_dist_metric)
ax.set_title("\n".join(wrap("Regression Error Characteristic from HLLE", 50)))
plt.savefig(output_dir + "rec_HLLE.svg")
plt.close(fig)

method_errs.pop("LTSA BPT")
method_errs.pop("LTSA Pruning")
from visualization.error_plots import relativeErrorBarChart
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
relativeErrorBarChart(ax, method_errs)
plt.savefig(output_dir + "reconstruction_error.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
listRegressionErrorCharacteristic(ax, embeddings_list, true_parameters, embeddings_name_list, dist_metric=err_dist_metric)
ax.set_title("\n".join(wrap("Regression Error Characteristic from All Methods", 50)))
plt.savefig(output_dir + "rec_combined.svg")
plt.close(fig)

write("Creating combined image...")
flush()
t0 = time.time()

matplotlib.rcParams.update({'font.size': 6})

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10.8, 10.8), dpi=100)
plt.tight_layout(pad=5, h_pad=10, w_pad=5)
axes_list = np.concatenate(axes)

for i in range(num_methods):
	solver = methods[i]
	name = method_names[i]
	feature_coords = solver.fit_transform(points)

	axes_list[i].scatter(feature_coords[:,0], feature_coords[:,1], c=color, cmap=plt.cm.Spectral, s=combined_sp_rad**2, linewidths=combined_sp_lw)
	axes_list[i].set_title("\n".join(wrap("2D Embedding from %s" % name, 25)))

feature_coords = compute_ltsa(points, neighbor_dict, observations, source_dim, target_dim)
axes_list[5].scatter(feature_coords[:,0], feature_coords[:,1], c=color, cmap=plt.cm.Spectral, s=combined_sp_rad**2, linewidths=combined_sp_lw)
axes_list[5].set_title("\n".join(wrap("2D Embedding from Classical LTSA", 25)))

feature_coords = compute_ltsa(points, neighbor_dict, mle_bases, source_dim, target_dim)
axes_list[6].scatter(feature_coords[:,0], feature_coords[:,1], c=color, cmap=plt.cm.Spectral, s=combined_sp_rad**2, linewidths=combined_sp_lw)
axes_list[6].set_title("\n".join(wrap("2D Embedding from LTSA with Tangent Space Correction", 25)))

feature_coords = compute_ltsa(points, pruned_neighbor_dict, mle_bases, source_dim, target_dim)
axes_list[7].scatter(feature_coords[:,0], feature_coords[:,1], c=color, cmap=plt.cm.Spectral, s=combined_sp_rad**2, linewidths=combined_sp_lw)
axes_list[7].set_title("\n".join(wrap("2D Embedding from LTSA with Tangent Space Correction and Edge Pruning", 25)))

# mds = MDS(n_components=target_dim, max_iter=3000, eps=1e-9, n_init=25, dissimilarity="precomputed", n_jobs=-1)
# feature_coords = mds.fit_transform(shortest_distances)
kpca = KernelPCA(n_components=target_dim, kernel="precomputed", eigen_solver=kpca_eigen_solver, tol=kpca_tol, max_iter=kpca_max_iter, n_jobs=-1)
feature_coords = kpca.fit_transform((shortest_distances**2) * -0.5)
axes_list[8].scatter(feature_coords[:,0], feature_coords[:,1], c=color, cmap=plt.cm.Spectral, s=combined_sp_rad**2, linewidths=combined_sp_lw)
axes_list[8].set_title("\n".join(wrap("2D Embedding from BP Tangent Correction for Edge Pruning", 25)))

plt.savefig(output_dir + "comparison_all.svg")
plt.close(fig)

t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

global_t1 = time.time()
write("\nTotal program runtime: %d seconds.\n\n" % (global_t1-global_t0))
flush()