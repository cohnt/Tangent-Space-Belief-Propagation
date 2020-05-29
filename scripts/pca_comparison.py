import numpy as np
import scipy
import random

import matplotlib.style
matplotlib.style.use('classic')

from textwrap import wrap
import matplotlib.pyplot as plt
import time
import sys
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import write, flush, pairwiseDistErr, setAxisTickSize
from collections import OrderedDict

global_t0 = time.time()

dataset_name = "long_spiral_curve"
# dataset_seed = 4045775215
# dataset_seed = 4015005259
dataset_seed = np.random.randint(0, 2**32)
num_points = 525    # Number of data points
data_noise = 0.0005 # How much noise is added to the data
num_outliers = 25
source_dim = 2      # The dimensionality of the incoming dataset (see "Load Dataset" below)
target_dim = 1      # The number of dimensions the data is being reduced to

num_iters = 25     # Number of iterations of the message passing algorithm to run
neighbors_k = 12    # The value of 'k' used for k-nearest-neighbors
num_samples = 10   # Numbers of samples used in the belief propagation algorithm
explore_perc = 0.1  # Fraction of uniform samples to keep exploring

message_resample_cov = np.eye(target_dim) * 0.01 # TODO: Change
pruning_angle_thresh = np.cos(30.0 * np.pi / 180.0)

error_histogram_num_bins = num_points / 10
err_dist_metric = "l2"
err_mat_norm = "max"

embedding_name = "KernelPCA" # Could also be MDS
kpca_eigen_solver = "auto"
kpca_tol = 1e-9
kpca_max_iter = 3000

data_sp_rad = 10.0
data_sp_lw = 1.0
nn_lw = 1.0
pca_ll = 0.1
embedding_sp_rad = 13.0
embedding_sp_lw = 1.0
combined_sp_rad = 4.0
combined_sp_lw = 0.5
embedding_axis_tick_size = 60
embedding_axis_n_ticks = 4
neighbors_axis_tick_size = 30
neighbors_axis_n_ticks = 7
embedding_axis_label_size = 30

################
# Load Dataset #
################
from datasets.dim_2.arc_curve import make_arc_curve
from datasets.dim_2.s_curve import make_s_curve
from datasets.dim_2.o_curve import make_o_curve
from datasets.dim_2.eight_curve import make_eight_curve
from datasets.dim_2.long_spiral_curve import make_long_spiral_curve

points, color, true_tangents, true_parameters, dataset_seed = make_long_spiral_curve(num_points, data_noise, rs_seed=dataset_seed)

mins = np.min(points, axis=0)
maxes = np.max(points, axis=0)
outliers = np.random.uniform(low=mins, high=maxes, size=(num_outliers, source_dim))
outlier_colors = np.zeros(num_outliers)
points[0:num_outliers] = outliers
color[0:num_outliers] = outlier_colors

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(points[:,0], points[:,1], c=color, cmap=plt.cm.Spectral, s=data_sp_rad**2, zorder=2, linewidth=data_sp_lw)
ax.scatter(points[0:num_outliers,0], points[0:num_outliers,1], color="black", s=data_sp_rad**2, linewidth=data_sp_lw, zorder=3)
ax.set_title("Dataset (num=%d, variance=%f, seed=%d)\n" % (num_points, data_noise, dataset_seed))
plt.show()

#######################
# k-Nearest-Neighbors #
#######################
from sklearn.neighbors import kneighbors_graph
from visualization.plot_neighbors import plot_neighbors_2d
from visualization.plot_pca import plot_pca_2d

neighbor_graph = kneighbors_graph(points, neighbors_k, mode="distance", n_jobs=-1)
# neighbor_graph is stored as a sparse matrix.
# Note that neighbor_graph is not necessarily symmetric, such as the case where point x
# is a nearest neighbor of point y, but point y is *not* a nearest neighbor of point x.
# We fix this later on...

####################
# Initialize Graph #
####################
from utils import sparseMatrixToDict, sparseMaximum

# Make the matrix symmetric by taking max(G, G^T)
neighbor_graph = sparseMaximum(neighbor_graph, neighbor_graph.T)
# This dictionary will have the structure point_idx: [list, of, neighbor_idx]
neighbor_dict = sparseMatrixToDict(neighbor_graph)
# This extracts all pairs of neighbors from the dictionary and stores them as a list of tuples.
# neighbor_pair_list represents the identification of the messages, i.e., "message 0" is
# so defined by being at index 0 of neighbor_pair_list.
neighbor_pair_list = [(key, value) for key, arr in neighbor_dict.items() for value in arr]
num_messages = len(neighbor_pair_list)

write("Number of points: %d\n" % num_points)
write("Number of edges: %d\n" % len(neighbor_pair_list))

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
plot_neighbors_2d(points, color, neighbor_graph, ax, point_size=data_sp_rad, line_width=data_sp_lw, edge_thickness=nn_lw, show_labels=False)
# ax.set_title("Nearest Neighbors (k=%d)\n" % neighbors_k)
setAxisTickSize(ax, neighbors_axis_tick_size, n_ticks=neighbors_axis_n_ticks)
ax.scatter(points[0:num_outliers,0], points[0:num_outliers,1], color="black", s=data_sp_rad**2, linewidth=data_sp_lw, zorder=3)
plt.show()

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
plot_pca_2d(points, color, true_tangents, ax, point_size=data_sp_rad, point_line_width=data_sp_lw, line_width=nn_lw, line_length=pca_ll)
ax.set_title("Exact Tangents")
plt.show()

###############
# Measure PCA #
###############
from sklearn.decomposition import PCA
from l1_pca import l1_pca

# n_components is the number of principal components pca will compute
pca = PCA(n_components=target_dim)
observations_l1 = [None for i in range(num_points)]
observations_l2 = [None for i in range(num_points)]
observations_leave_one_out = [None for i in range(num_points)]

write("Computing PCA observations...")
flush()
t0 = time.time()
for i in range(num_points):
	print i
	og_point = points[i]
	row = neighbor_graph.toarray()[i]
	neighbors = np.nonzero(row)[0]
	neighborhood = points[neighbors]
	observations_l1[i] = l1_pca(neighborhood.T, target_dim).T
	pca.fit(neighborhood)
	# vec1 = pca.components_[0]
	observations_l2[i] = pca.components_[0:target_dim]
	#
	vec_list = neighborhood - og_point
	dist_list = np.linalg.norm(vec_list, axis=1)
	max_dist_ind = np.argmax(dist_list)
	new_neighborhood = np.delete(neighborhood, max_dist_ind, axis=0)
	pca.fit(new_neighborhood)
	observations_leave_one_out[i] = pca.components_[0:target_dim]
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
# plot_neighbors_2d(points, color, neighbor_graph, ax, point_size=2, line_width=0.25, edge_thickness=0.1, show_labels=False)
plot_pca_2d(points, color, observations_l1, ax, point_size=data_sp_rad, point_line_width=data_sp_lw, line_width=nn_lw, line_length=pca_ll)
ax.scatter(points[0:num_outliers,0], points[0:num_outliers,1], color="black", s=data_sp_rad**2, linewidth=data_sp_lw, zorder=3)
ax.set_title("Measured Tangent Spaces (L1-PCA)")
plt.show()

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
# plot_neighbors_2d(points, color, neighbor_graph, ax, point_size=2, line_width=0.25, edge_thickness=0.1, show_labels=False)
plot_pca_2d(points, color, observations_l2, ax, point_size=data_sp_rad, point_line_width=data_sp_lw, line_width=nn_lw, line_length=pca_ll)
ax.scatter(points[0:num_outliers,0], points[0:num_outliers,1], color="black", s=data_sp_rad**2, linewidth=data_sp_lw, zorder=3)
ax.set_title("Measured Tangent Spaces (L2-PCA)")
plt.show()

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
# plot_neighbors_2d(points, color, neighbor_graph, ax, point_size=2, line_width=0.25, edge_thickness=0.1, show_labels=False)
plot_pca_2d(points, color, observations_leave_one_out, ax, point_size=data_sp_rad, point_line_width=data_sp_lw, line_width=nn_lw, line_length=pca_ll)
ax.scatter(points[0:num_outliers,0], points[0:num_outliers,1], color="black", s=data_sp_rad**2, linewidth=data_sp_lw, zorder=3)
ax.set_title("Measured Tangent Spaces (L2-PCA, ignore furthest point)")
plt.show()
