import numpy as np
import scipy
import random

import matplotlib
matplotlib.use('Agg')
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
dataset_seed = 4045775215
# dataset_seed = 4015005259
# dataset_seed = np.random.randint(0, 2**32)
num_points = 525    # Number of data points
data_noise = 0.0005 # How much noise is added to the data
num_outliers = 25
source_dim = 2      # The dimensionality of the incoming dataset (see "Load Dataset" below)
target_dim = 1      # The number of dimensions the data is being reduced to
neighbors_k = 12    # The value of 'k' used for k-nearest-neighbors

data_sp_rad = 10.0
data_sp_lw = 1.0
nn_lw = 1.0
pca_ll = 0.1

output_dir = ""

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

mins = np.min(points, axis=0)
maxes = np.max(points, axis=0)
outliers = np.random.uniform(low=mins, high=maxes, size=(num_outliers, source_dim))
outlier_colors = np.zeros(num_outliers)
points[0:num_outliers] = outliers
color[0:num_outliers] = outlier_colors

#######################
# k-Nearest-Neighbors #
#######################
from sklearn.neighbors import kneighbors_graph
from visualization.plot_neighbors import plot_neighbors_2d
from visualization.plot_pca import plot_pca_2d

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
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
plot_neighbors_2d(points, color, neighbor_graph, ax, point_size=data_sp_rad, line_width=data_sp_lw, edge_thickness=nn_lw, show_labels=False)
# ax.set_title("Nearest Neighbors (k=%d)\n" % neighbors_k)
ax.scatter(points[0:num_outliers,0], points[0:num_outliers,1], color="black", s=data_sp_rad**2, linewidth=data_sp_lw, zorder=3)
plt.savefig(output_dir + "nearest_neighbors.svg")
plt.close(fig)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

write("Saving ground truth tangent plot...")
flush()
t0 = time.time()
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
plot_pca_2d(points, color, true_tangents, ax, point_size=data_sp_rad, point_line_width=data_sp_lw, line_width=nn_lw, line_length=pca_ll)
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
from l1_pca import l1_pca

# n_components is the number of principal components pca will compute
pca = PCA(n_components=target_dim)
observations_l1 = [None for i in range(num_points)]
observations_l2 = [None for i in range(num_points)]

write("Computing PCA observations...")
flush()
t0 = time.time()
for i in range(num_points):
	og_point = points[i]
	row = neighbor_graph.toarray()[i]
	neighbors = np.nonzero(row)[0]
	neighborhood = points[neighbors]
	
	print "Computing L1 PCA for observation %d" % i
	observations_l1[i] = l1_pca(neighborhood.T, target_dim).T

	pca.fit(neighborhood)
	# vec1 = pca.components_[0]
	observations_l2[i] = pca.components_[0:target_dim]
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

write("Saving PCA observations plots...")
flush()
t0 = time.time()
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
# plot_neighbors_2d(points, color, neighbor_graph, ax, point_size=2, line_width=0.25, edge_thickness=0.1, show_labels=False)
plot_pca_2d(points, color, observations_l1, ax, point_size=data_sp_rad, point_line_width=data_sp_lw, line_width=nn_lw, line_length=pca_ll)
ax.scatter(points[0:num_outliers,0], points[0:num_outliers,1], color="black", s=data_sp_rad**2, linewidth=data_sp_lw, zorder=3)
ax.set_title("Measured Tangent Spaces (PCA)")
plt.savefig(output_dir + "l1_pca_observations.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
# plot_neighbors_2d(points, color, neighbor_graph, ax, point_size=2, line_width=0.25, edge_thickness=0.1, show_labels=False)
plot_pca_2d(points, color, observations_l2, ax, point_size=data_sp_rad, point_line_width=data_sp_lw, line_width=nn_lw, line_length=pca_ll)
ax.scatter(points[0:num_outliers,0], points[0:num_outliers,1], color="black", s=data_sp_rad**2, linewidth=data_sp_lw, zorder=3)
ax.set_title("Measured Tangent Spaces (PCA)")
plt.savefig(output_dir + "l2_pca_observations.svg")
plt.close(fig)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()