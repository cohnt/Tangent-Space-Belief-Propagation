import numpy as np
import scipy
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.style
matplotlib.style.use('classic')

import matplotlib.pyplot as plt
from collections import OrderedDict

from utils import setAxisTickSize

neighbors_k = 10

output_dir = "results_landmark/"

embedding_point_radius = 13.0
embedding_axis_tick_size = 40
neighbors_axis_tick_size = 30
title_font_size = 30

matplotlib.rcParams.update({'font.size': 24})

# landmark_coords = [
# 	np.array([0., 0.]),
# 	np.array([5., 5.]),
# 	np.array([0.,10.])
# ]
# num_landmarks = len(landmark_coords)

# landmark_coords = []
# num_landmarks = 4
# for _ in range(num_landmarks):
# 	landmark_coords.append(np.random.uniform(low=0.0, high=10.0, size=(2)))

# landmark_coords = [
# 	np.array([6.93128243, 2.88532531]),
# 	np.array([3.54094086, 5.58023536]),
# 	np.array([1.46209511, 9.62733474])
# ]
# num_landmarks = len(landmark_coords)

landmark_coords = [
	np.array([6.93128243, 2.88532531]),
	np.array([3.54094086, 5.58023536]),
	np.array([1.46209511, 10.62733474])
]
num_landmarks = len(landmark_coords)

# landmark_coords = [
# 	np.array([6.98200123, 6.30701217]),
# 	np.array([6.72636139, 6.84291166]),
# 	np.array([3.45049685, 6.74195935]),
# 	np.array([9.48523194, 5.73978978])
# ]
# num_landmarks = len(landmark_coords)

# landmark_coords = [
# 	np.array([0.44181691, 2.47556252]),
# 	np.array([9.74855701, 8.67657555]),
# 	np.array([0.94111181, 3.8372051 ]),
# 	np.array([4.20762079, 5.0563642 ])
# ]
# num_landmarks = len(landmark_coords)

# landmark_coords = [
# 	np.array([5.27770294, 4.54453647]),
# 	np.array([6.16563489, 5.39577828]),
# 	np.array([7.16979217, 9.11084062]),
# 	np.array([7.29313071, 9.15263291])
# ]
# num_landmarks = len(landmark_coords)

# landmark_coords = np.append(np.stack((np.linspace(1, 1, num=10), np.linspace(1, 10, num=10)), axis=-1), np.stack((np.linspace(10, 10, num=10), np.linspace(1, 10, num=10)), axis=-1), axis=0)
# landmark_coords = landmark_coords + np.random.uniform(low=-0.1, high=0.1, size=landmark_coords.shape)
# num_landmarks = len(landmark_coords)

# landmark_coords = np.stack((np.linspace(1, 10, num=40), np.linspace(1, 10, num=40)), axis=-1)
# landmark_coords = landmark_coords + np.random.uniform(low=-0.75, high=0.75, size=landmark_coords.shape)
# num_landmarks = len(landmark_coords)

print "Landmarks:"
for landmark in landmark_coords:
	print landmark

# Build up a list of points to measure at
x_vals = np.linspace(-1.5, 11.5, num=31)
y_vals = np.linspace(-1.5, 11.5, num=31)
xx, yy = np.meshgrid(x_vals, y_vals)
points = np.stack((np.ravel(xx), np.ravel(yy)), axis=-1)
# points = points[np.random.choice(range(len(points)), 400, replace=False)]
mask = np.logical_or(np.logical_or(points[:,1] > 7, points[:,1] < 3), np.logical_or(points[:,0] > 8, points[:,0] < 2))
points = points[mask]
num_points = len(points)

range_data = np.zeros((num_points, num_landmarks))

def noise():
	# return np.random.uniform(low=-0.5, high=0.5)
	return np.random.normal(loc=0.0, scale=0.0)

for i in range(num_points):
	for j in range(num_landmarks):
		range_data[i][j] = np.linalg.norm(points[i] - landmark_coords[j]) + noise()

################

from sklearn.neighbors import kneighbors_graph
from visualization.plot_neighbors import plot_neighbors_2d
from utils import sparseMatrixToDict, sparseMaximum

neighbor_graph = kneighbors_graph(range_data, neighbors_k, mode="distance", n_jobs=-1)
neighbor_graph = sparseMaximum(neighbor_graph, neighbor_graph.T)
neighbor_dict = sparseMatrixToDict(neighbor_graph)
neighbor_pair_list = [(key, value) for key, arr in neighbor_dict.items() for value in arr]

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
plot_neighbors_2d(points, points[:,0]/10.0, neighbor_graph, ax, show_labels=False, point_size=embedding_point_radius)
ax.scatter(np.asarray(landmark_coords)[:,0], np.asarray(landmark_coords)[:,1], c="black", s=(embedding_point_radius*3)**2, marker="*", zorder=100)
ax.set_title("Nearest Neighbors (k=%d)\n" % neighbors_k, fontsize=title_font_size)
setAxisTickSize(ax, neighbors_axis_tick_size)
plt.savefig(output_dir + "nearest_neighbors.svg")
plt.close(fig)

#################





from utils import write, flush
import scipy
import random
import matplotlib
from textwrap import wrap
import time
import sys
import copy
from joblib import Parallel, delayed
from tqdm import tqdm

target_dim = 2
source_dim = num_landmarks

num_samples = 5
num_iters = 25

explore_perc = 0

message_resample_cov = np.eye(target_dim) * 0.01 # TODO: Change
pruning_angle_thresh = 0.98
ts_noise_variance = 0.01 # In degrees

embedding_name = "KernelPCA" # Could also be MDS
kpca_eigen_solver = "auto"
kpca_tol = 1e-9
kpca_max_iter = 3000


true_vals = points.copy()
points = range_data.copy()


##########

# COMPARISONS

from utils import pairwiseDistErr
from visualization.error_plots import regressionErrorCharacteristic, listRegressionErrorCharacteristic

from sklearn.manifold import LocallyLinearEmbedding, MDS, Isomap, SpectralEmbedding, TSNE
from ltsa import compute_ltsa
from autoencoder import Autoencoder

methods = []
methods.append(LocallyLinearEmbedding(n_neighbors=neighbors_k, n_components=target_dim, n_jobs=-1))
methods.append(MDS(n_components=target_dim, n_jobs=-1))
methods.append(Isomap(n_neighbors=neighbors_k, n_components=target_dim, n_jobs=-1))
methods.append(SpectralEmbedding(n_components=target_dim, n_neighbors=neighbors_k, n_jobs=-1))
methods.append(TSNE(n_components=target_dim))
methods.append(LocallyLinearEmbedding(n_neighbors=neighbors_k, n_components=target_dim, n_jobs=-1, method="ltsa"))
methods.append(Autoencoder(source_dim, target_dim, [64, 32, 32], ["relu", "relu", "relu"]))
num_methods = len(methods)

method_names = ["LLE", "MDS", "Isomap", "SpectralEmbedding", "t-SNE", "LTSA", "Autoencoder"]

embeddings_list = []
embeddings_name_list = []

rec_max_errors = OrderedDict()
rec_mean_errors = OrderedDict()
rec_fro_errors = OrderedDict()

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

	embeddings_list.append(feature_coords)
	embeddings_name_list.append(name)

	fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
	ax.scatter(feature_coords[:,0], feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
	setAxisTickSize(ax, embedding_axis_tick_size)
	plt.savefig(output_dir + "embedding_%s_1_x.svg" % name)
	plt.close(fig)
	fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
	ax.scatter(feature_coords[:,0], feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
	setAxisTickSize(ax, embedding_axis_tick_size)
	plt.savefig(output_dir + "embedding_%s_1_y.svg" % name)
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
	ax.scatter(feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
	setAxisTickSize(ax, embedding_axis_tick_size)
	plt.savefig(output_dir + "embedding_%s_2_x.svg" % name)
	plt.close(fig)
	fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
	ax.scatter(feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
	setAxisTickSize(ax, embedding_axis_tick_size)
	plt.savefig(output_dir + "embedding_%s_2_y.svg" % name)
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
	ax.scatter(-feature_coords[:,0], feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
	setAxisTickSize(ax, embedding_axis_tick_size)
	plt.savefig(output_dir + "embedding_%s_3_x.svg" % name)
	plt.close(fig)
	fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
	ax.scatter(-feature_coords[:,0], feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
	setAxisTickSize(ax, embedding_axis_tick_size)
	plt.savefig(output_dir + "embedding_%s_3_y.svg" % name)
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
	ax.scatter(-feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
	setAxisTickSize(ax, embedding_axis_tick_size)
	plt.savefig(output_dir + "embedding_%s_4_x.svg" % name)
	plt.close(fig)
	fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
	ax.scatter(-feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
	setAxisTickSize(ax, embedding_axis_tick_size)
	plt.savefig(output_dir + "embedding_%s_4_y.svg" % name)
	plt.close(fig)

	print "%s max error: %f" % (name, pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="max"))
	print "%s avg error: %f" % (name, pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="mean"))
	print "%s med error: %f" % (name, pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="median"))
	print "%s fro error: %f" % (name, pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="fro"))

	rec_max_errors[name] = pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="max")
	rec_mean_errors[name] = pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="mean")
	rec_fro_errors[name] = pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="fro")

############

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(true_vals[:,0], true_vals[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "ideal_embedding_x.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(true_vals[:,0], true_vals[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "ideal_embedding_y.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(true_vals[:,0], true_vals[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
ax.scatter(np.asarray(landmark_coords)[:,0], np.asarray(landmark_coords)[:,1], c="black", s=(embedding_point_radius*3)**2, marker="*")
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "layout_x.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(true_vals[:,0], true_vals[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
ax.scatter(np.asarray(landmark_coords)[:,0], np.asarray(landmark_coords)[:,1], c="black", s=(embedding_point_radius*3)**2, marker="*")
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "layout_y.svg")
plt.close(fig)

############################
# Write Current Code State #
############################
import inspect
import sys

lines = inspect.getsourcelines(sys.modules[__name__])[0]

f = open(output_dir + "code.py", "w")
for index, line in enumerate(lines):
	f.write("{:4d} {}".format(index + 1, line))

f.close()

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

def randomSmallRotation(dimension, variance=None):
	if variance is None:
		variance = 0.05 * dimension * 180.0 / np.pi
	theta = np.random.normal(0, variance) * np.pi / 180.0
	rotMat = np.eye(dimension)
	rotMat[0,0] = np.cos(theta)
	rotMat[0,1] = -np.sin(theta)
	rotMat[1,0] = np.sin(theta)
	rotMat[1,1] = np.cos(theta)
	# basis = special_ortho_group.rvs(dimension)
	# basis_inv = basis.transpose()
	# return basis.dot(rotMat).dot(basis_inv)
	return np.eye(dimension)

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
	return np.array([np.dot(rotMat, ts[0])])
	# theta = np.random.normal(0, var) * np.pi / 180.0
	# rotMat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
	# noise_mat = np.random.normal(0, ts_noise_variance, ts.shape)
	# ts = ts + noise_mat
	# ts = scipy.linalg.orth(ts.T).T
	# return np.array(ts)

def noisifyTSList(ts_list, var=5):
	for i in range(len(ts_list)):
		ts_list[i] = noisifyTS(ts_list[i], var)
	return ts_list

# This initializes messages_prev and messages_next as num_points by num_points arrays of Nones.
# Where appropriate, the Nones will be replaced by Message objects
messages_prev = [[None for __ in range(num_points)] for _ in range(num_points)]
messages_next = [[None for __ in range(num_points)] for _ in range(num_points)]
# for i in tqdm(range(len(neighbor_pair_list))):
# 	key, value = neighbor_pair_list[i]
for key, value in neighbor_pair_list:
	# Note that key represents where the message is coming from and value represents where the message is going to
	# In other words, messages[key][value] === m_key->value
	messages_prev[key][value] = Message(num_samples, source_dim, target_dim)
	# messages_prev[key][value].ts = randomTangentSpaceList(num_samples, source_dim, target_dim)
	messages_prev[key][value].ts = np.repeat([observations[value]], num_samples, axis=0)
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

		total_time = message_time + belief_time + image_time + graph_time
		write("Total iteration time: %f\n" % total_time)
except KeyboardInterrupt:
	write("\nTerminating early after %d iterations.\n" % (iter_num-1))
	write("Iteration %d not completed.\n\n" % iter_num)
	flush()

write("Pruning edges...")
flush()
t0 = time.time()

mle_bases = np.zeros((num_points, target_dim, source_dim))
for i in range(num_points):
	max_ind = np.argmax(belief[i].weights)
	mle_bases[i] = belief[i].ts[max_ind]

num_pruned = 0
pruned_neighbors = scipy.sparse.lil_matrix(neighbor_graph)
for i in range(0, num_points):
	for j in range(0, num_points):
		if i != j:
			vec = points[j] - points[i]
			proj_vec = projSubspace(mle_bases[i], vec)
			dprod = np.abs(np.dot(vec, np.dot(proj_vec, mle_bases[i])) / (np.linalg.norm(vec) * np.linalg.norm(proj_vec)))
			# print dprod
			if dprod < pruning_angle_thresh:
				pruned_neighbors[i,j] = 0
				pruned_neighbors[j,i] = 0
				num_pruned = num_pruned + 1

t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

print "Pruned %d edges!" % num_pruned

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
plot_neighbors_2d(true_vals, true_vals[:,0]/10.0, pruned_neighbors, ax, show_labels=False, point_size=embedding_point_radius)
ax.scatter(np.asarray(landmark_coords)[:,0], np.asarray(landmark_coords)[:,1], c="black", s=(embedding_point_radius*3)**2, marker="*", zorder=100)
ax.set_title("Pruned Nearest Neighbors\n", fontsize=title_font_size)
setAxisTickSize(ax, neighbors_axis_tick_size)
plt.savefig(output_dir + "pruned_neighbors.svg")
plt.close(fig)

write("Connecting graph...\n")
flush()
t0 = time.time()

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
	while len(connected_components.sets) > 1:
		write("There are %d connected components.\n" % len(connected_components.sets))
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

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
plot_neighbors_2d(true_vals, true_vals[:,0]/10.0, pruned_neighbors, ax, show_labels=False, point_size=embedding_point_radius)
ax.set_title("Reconnected Nearest Neighbors\n", fontsize=title_font_size)
setAxisTickSize(ax, neighbors_axis_tick_size)
plt.savefig(output_dir + "added_edges.svg")
plt.close(fig)

from sklearn.utils.graph_shortest_path import graph_shortest_path
write("Finding shortest paths...")
flush()
t0 = time.time()
shortest_distances = graph_shortest_path(pruned_neighbors, directed=False)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

###################################

solver = TSNE(n_components=target_dim, metric="precomputed")
feature_coords = solver.fit_transform(graph_shortest_path(neighbor_graph, directed=False))

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_NN_t-SNE_1_x.svg")
plt.close(fig)
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_NN_t-SNE_1_y.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_NN_t-SNE_2_x.svg")
plt.close(fig)
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_NN_t-SNE_2_y.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(-feature_coords[:,0], feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_NN_t-SNE_3_x.svg")
plt.close(fig)
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(-feature_coords[:,0], feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_NN_t-SNE_3_y.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(-feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_NN_t-SNE_4_x.svg")
plt.close(fig)
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(-feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_NN_t-SNE_4_y.svg")
plt.close(fig)

print "NN t-SNE max error: %f" % pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="max")
print "NN t-SNE avg error: %f" % pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="mean")
print "NN t-SNE med error: %f" % pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="median")
print "NN t-SNE fro error: %f" % pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="fro")

###################################

solver = TSNE(n_components=target_dim, metric="precomputed")
feature_coords = solver.fit_transform(shortest_distances)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_Corrected_t-SNE_1_x.svg")
plt.close(fig)
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_Corrected_t-SNE_1_y.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_Corrected_t-SNE_2_x.svg")
plt.close(fig)
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_Corrected_t-SNE_2_y.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(-feature_coords[:,0], feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_Corrected_t-SNE_3_x.svg")
plt.close(fig)
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(-feature_coords[:,0], feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_Corrected_t-SNE_3_y.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(-feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_Corrected_t-SNE_4_x.svg")
plt.close(fig)
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(-feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_Corrected_t-SNE_4_y.svg")
plt.close(fig)

print "Corrected t-SNE max error: %f" % pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="max")
print "Corrected t-SNE avg error: %f" % pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="mean")
print "Corrected t-SNE med error: %f" % pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="median")
print "Corrected t-SNE fro error: %f" % pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="fro")

###################################

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

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_tsbp_1_x.svg")
plt.close(fig)
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_tsbp_1_y.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_tsbp_2_x.svg")
plt.close(fig)
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_tsbp_2_y.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(-feature_coords[:,0], feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_tsbp_3_x.svg")
plt.close(fig)
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(-feature_coords[:,0], feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_tsbp_3_y.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(-feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_tsbp_4_x.svg")
plt.close(fig)
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(-feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "embedding_tsbp_4_y.svg")
plt.close(fig)

# tsbp_error = pairwiseDistErr(feature_coords, points, dist_metric="l2", mat_norm="max")
# print "TSBP error: %f" % tsbp_error
print "TSBP max error: %f" % pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="max")
print "TSBP avg error: %f" % pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="mean")
print "TSBP med error: %f" % pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="median")
print "TSBP fro error: %f" % pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="fro")

def ordered_dict_prepend(dct, key, value, dict_setitem=dict.__setitem__):
	root = dct._OrderedDict__root
	first = root[1]
	if key in dct:
		link = dct._OrderedDict__map[key]
		link_prev, link_next, _ = link
		link_prev[1] = link_next
		link_next[0] = link_prev
		link[0] = root
		link[1] = first
		root[1] = first[0] = link
	else:
		root[1] = first[0] = dct._OrderedDict__map[key] = [root, first, key]
		dict_setitem(dct, key, value)

ordered_dict_prepend(rec_max_errors, "TSBP", pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="max"))
ordered_dict_prepend(rec_mean_errors, "TSBP", pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="mean"))
ordered_dict_prepend(rec_fro_errors, "TSBP", pairwiseDistErr(feature_coords, true_vals, dist_metric="l2", mat_norm="fro"))

embeddings_list.insert(0, feature_coords)
embeddings_name_list.insert(0, "TSBP")

##################

fig, ax = plt.subplots()
listRegressionErrorCharacteristic(ax, embeddings_list, true_vals, embeddings_name_list, dist_metric="l2")
setAxisTickSize(ax, 15)
ax.xaxis.label.set_size(20)
ax.yaxis.label.set_size(20)
plt.savefig(output_dir + "rec.svg")
plt.close(fig)

##################

from visualization.error_plots import relativeErrorBarChart

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
relativeErrorBarChart(ax, rec_max_errors, title="Maximum Pairwise Error by Manifold Learning Algorithm\n", titlefontsize=title_font_size)
setAxisTickSize(ax, 20)
ax.yaxis.label.set_size(20)
plt.savefig(output_dir + "reconstruction_error_max.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
relativeErrorBarChart(ax, rec_mean_errors, title="Mean Pairwise Error by Manifold Learning Algorithm\n", titlefontsize=title_font_size)
setAxisTickSize(ax, 20)
ax.yaxis.label.set_size(20)
plt.savefig(output_dir + "reconstruction_error_mean.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
relativeErrorBarChart(ax, rec_fro_errors, title="Frobenius Reconstruction Error by Manifold Learning Algorithm\n", titlefontsize=title_font_size)
setAxisTickSize(ax, 20)
ax.yaxis.label.set_size(20)
plt.savefig(output_dir + "reconstruction_error_fro.svg")
plt.close(fig)

##################

ideal_neighbor_graph = kneighbors_graph(true_vals, neighbors_k, mode="distance", n_jobs=-1)
ideal_shortest_distances = graph_shortest_path(ideal_neighbor_graph, directed=False)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
plot_neighbors_2d(true_vals, true_vals[:,0]/10.0, ideal_neighbor_graph, ax, show_labels=False, point_size=embedding_point_radius)
ax.set_title("Ideal Nearest Neighbors (k=%d)\n" % neighbors_k, fontsize=title_font_size)
setAxisTickSize(ax, neighbors_axis_tick_size)
plt.savefig(output_dir + "ideal_nearest_neighbors.svg")
plt.close(fig)

# Ideal ISOMAP

kpca = KernelPCA(n_components=target_dim, kernel="precomputed", eigen_solver=kpca_eigen_solver, tol=kpca_tol, max_iter=kpca_max_iter, n_jobs=-1)
feature_coords = kpca.fit_transform((ideal_shortest_distances**2) * -0.5)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "ideal_isomap_1_x.svg")
plt.close(fig)
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "ideal_isomap_1_y.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "ideal_isomap_2_x.svg")
plt.close(fig)
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "ideal_isomap_2_y.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(-feature_coords[:,0], feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "ideal_isomap_3_x.svg")
plt.close(fig)
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(-feature_coords[:,0], feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "ideal_isomap_3_y.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(-feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "ideal_isomap_4_x.svg")
plt.close(fig)
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(-feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "ideal_isomap_4_y.svg")
plt.close(fig)

# Ideal t-SNE

solver = TSNE(n_components=target_dim, metric="precomputed")
feature_coords = solver.fit_transform(ideal_shortest_distances)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "ideal_tsne_1_x.svg")
plt.close(fig)
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "ideal_tsne_1_y.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "ideal_tsne_2_x.svg")
plt.close(fig)
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "ideal_tsne_2_y.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(-feature_coords[:,0], feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "ideal_tsne_3_x.svg")
plt.close(fig)
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(-feature_coords[:,0], feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "ideal_tsne_3_y.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(-feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,0]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "ideal_tsne_4_x.svg")
plt.close(fig)
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(-feature_coords[:,0], -feature_coords[:,1], c=true_vals[:,1]/10.0, cmap=plt.cm.Spectral, s=embedding_point_radius**2)
setAxisTickSize(ax, embedding_axis_tick_size)
plt.savefig(output_dir + "ideal_tsne_4_y.svg")
plt.close(fig)