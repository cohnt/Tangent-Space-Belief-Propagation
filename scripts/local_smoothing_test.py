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
from utils import write, flush

dataset_name = "long_spiral_curve"
dataset_seed = np.random.randint(2**32)
num_points = 500    # Number of data points
data_noise = 0.001 # How much noise is added to the data
source_dim = 2      # The dimensionality of the incoming dataset (see "Load Dataset" below)
target_dim = 1      # The number of dimensions the data is being reduced to

neighbors_k = 12    # The value of 'k' used for k-nearest-neighbors
num_iters = 100
eps = 10e-12
gamma = 1.0

output_dir = "other_results/results_local_smoothing/"

write("\n")

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

f.write("\n[Smoothing]\n")
f.write("num_neighbors=%d\n" % neighbors_k)
f.write("max_iters=%d\n" % num_iters)
f.write("epsilon=%f\n" % eps)
f.write("gamma=%f\n" % gamma)

f.write("\n[Display]\n")
f.write("; TODO\n")

f.close()

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

write("Saving dataset plot...")
flush()
t0 = time.time()
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(points[:,0], points[:,1], c=color, cmap=plt.cm.Spectral, s=2**2, zorder=2, linewidth=0.25)
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

write("Saving nearest neighbors plot...")
flush()
t0 = time.time()
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
plot_neighbors_2d(points, color, neighbor_graph, ax, point_size=2, line_width=0.25, edge_thickness=0.5, show_labels=False)
ax.set_title("Nearest Neighbors (k=%d)" % neighbors_k)
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

write("Number of points: %d\n" % num_points)
write("Number of edges: %d\n" % len(neighbor_pair_list))

#############################
# Computing Local Smoothing #
#############################

smoothed = []

for i in range(num_points):
	print "i", i
	nbd_idx = np.append(neighbor_dict[i], i)
	print "nbd_idx", nbd_idx
	print "shape", nbd_idx.shape
	nbd = points[nbd_idx]
	weights = np.zeros(len(nbd))
	x_mean = np.mean(nbd, axis=0)
	print "Testing point", i, points[i]
	print "Initial mean", x_mean
	for iter_num in range(num_iters):
		old_mean = x_mean.copy()

		# Weights update
		for j in range(len(nbd_idx)):
			point = points[nbd_idx[j]]
			weights[j] = np.exp(-1.0 * gamma * (np.linalg.norm(point - x_mean)**2))
		total = np.sum(weights)
		weights = weights / total

		# Weighted mean update
		x_mean = np.average(nbd, axis=0, weights=weights)
		diff = np.linalg.norm(x_mean-old_mean)
		print "Old:", old_mean, "\t\tNew:", x_mean, "\t\tDiff: ", diff

		if diff < eps:
			print "Converged!\n"
			break
	smoothed.append(x_mean.copy())

smoothed = np.asarray(smoothed)

#######################
# Save Smoothed Graph #
#######################

write("Saving smoothed dataset plot...")
flush()
t0 = time.time()
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(smoothed[:,0], smoothed[:,1], c=color, cmap=plt.cm.Spectral, s=2**2, zorder=2, linewidth=0.25)
ax.set_title("Smoothed (epsilon=%f, gamma=%f)" % (eps, gamma))
plt.savefig(output_dir + "smoothed.svg")
plt.close(fig)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

###########################
# Nearest-Neighbors Again #
###########################

write("Computing nearest neighbors again...")
flush()
t0 = time.time()
neighbor_graph = kneighbors_graph(smoothed, neighbors_k, mode="distance", n_jobs=-1)
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
fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
plot_neighbors_2d(smoothed, color, neighbor_graph, ax, point_size=2, line_width=0.25, edge_thickness=0.5, show_labels=False)
ax.set_title("Nearest Neighbors (k=%d)" % neighbors_k)
plt.savefig(output_dir + "smoothed_neighbors.svg")
plt.close(fig)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()