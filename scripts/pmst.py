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
r = 1.0
T = 10
gamma = 1.0

output_dir = "results_mst/"

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

f.write("\n[PMST]\n")
f.write("num_neighbors=%d\n" % neighbors_k)
f.write("r=%f\n" % r)
f.write("T=%d\n" % T)

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
points, color, true_tangents, dataset_seed = make_long_spiral_curve(num_points, data_noise, rs_seed=dataset_seed)
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

######################
# Perturbing Dataset #
######################

dataset_list = []
for _ in range(T):
	points_cpy = points.copy()
	for i in range(num_points):
		nbd_idx = np.append(neighbor_dict[i], i)
		nbd = points[nbd_idx]
		dists = np.zeros(len(nbd))
		for j in range(len(nbd_idx)):
			neighbor = points_cpy[nbd_idx[j]]
			dists[j] = np.linalg.norm(points_cpy[i] - neighbor)
		di = np.mean(dists)
		stdev = di * r
		bound = np.sqrt(12) * stdev * 0.5
		rand_noise = np.random.uniform(-bound, bound, size=source_dim)
		points_cpy[i] = points_cpy[i] + rand_noise
	dataset_list.append(points_cpy.copy())

for i in range(T):
	print "Saving perturbed dataset %d" % i
	fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
	ax.scatter(dataset_list[i][:,0], dataset_list[i][:,1], c=color, cmap=plt.cm.Spectral, s=2**2, zorder=2, linewidth=0.25)
	ax.set_title("Perturbed Dataset %d" % i)
	plt.savefig(output_dir + "perturbed_%d.svg" % i)
	plt.close(fig)
