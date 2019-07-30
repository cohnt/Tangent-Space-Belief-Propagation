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

global_t0 = time.time()

dataset_name = "helix_curve"
dataset_seed = np.random.randint(0, 2**32)
num_points = 500    # Number of data points
data_noise = 0      # How much noise is added to the data
source_dim = 3      # The dimensionality of the incoming dataset (see "Load Dataset" below)
target_dim = 1      # The number of dimensions the data is being reduced to

num_iters = 10      # Number of iterations of the message passing algorithm to run
neighbors_k = 12    # The value of 'k' used for k-nearest-neighbors
num_samples = 5     # Numbers of samples used in the belief propagation algorithm
explore_perc = 0.1  # Fraction of uniform samples to keep exploring

message_resample_cov = np.eye(target_dim) * 0.01 # TODO: Change
pruning_angle_thresh = np.cos(30.0 * np.pi / 180.0)

output_dir = "results_helix/"
error_histogram_num_bins = num_points / 10

embedding_name = "KernelPCA" # Could also be MDS
kpca_eigen_solver = "auto"
kpca_tol = 1e-9
kpca_max_iter = 3000

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

f.write("\n[Belief Propagation]\n")
f.write("max_iters=%d\n" % num_iters)
f.write("num_neighbors=%d\n" % neighbors_k)
f.write("num_samples=%d\n" % num_samples)
f.write("explore=%s\n" % str(explore_perc))
f.write("prune_thresh=%s\n" % str(pruning_angle_thresh))

f.write("\n[Embedding]\n")
f.write("embedding_method=%s\n" % embedding_name)
f.write("embedding_eigen_solver=%s\n" % kpca_eigen_solver)
f.write("embedding_tol=%s\n" % str(kpca_tol))
f.write("embedding_max_iter=%d\n" % kpca_max_iter)

f.write("\n[Display]\n")
f.write("; TODO\n")

f.close()

################
# Load Dataset #
################
from datasets.dim_3.helix_curve import make_helix_curve
from mpl_toolkits.mplot3d import Axes3D

write("Generating dataset...")
flush()
t0 = time.time()
points, color, true_tangents, dataset_seed = make_helix_curve(num_points, data_noise, rs_seed=dataset_seed)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

write("Saving dataset plot...")
flush()
t0 = time.time()
fig = plt.figure(figsize=(14.4, 10.8), dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2], c=color, cmap=plt.cm.Spectral, s=2**2, zorder=2, linewidth=0.25)
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
fig = plt.figure(figsize=(14.4, 10.8), dpi=100)
ax = fig.add_subplot(111, projection='3d')
plot_neighbors_3d(points, color, neighbor_graph, ax, point_size=2, line_width=0.25, edge_thickness=0.5, show_labels=False)
ax.set_title("Nearest Neighbors (k=%d)" % neighbors_k)
plt.savefig(output_dir + "nearest_neighbors.svg")
plt.close(fig)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

write("Saving ground truth tangent plot...")
flush()
t0 = time.time()
fig = plt.figure(figsize=(14.4, 10.8), dpi=100)
ax = fig.add_subplot(111, projection='3d')
plot_pca_3d(points, color, true_tangents, ax, point_size=2, point_line_width=0.25, line_width=0.5, line_length=0.05)
ax.set_title("Exact Tangents")
plt.savefig(output_dir + "true_tangents.svg")
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
fig = plt.figure(figsize=(14.4, 10.8), dpi=100)
ax = fig.add_subplot(111, projection='3d')
plot_pca_3d(points, color, observations, ax, point_size=2, point_line_width=0.25, line_width=0.5, line_length=0.05)
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
	rotMat = randomSmallRotation(3, variance=var)
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
	messages_prev[key][value].ts = noisifyTSList(np.repeat([observations[value]], num_samples, axis=0), var=30)
	messages_prev[key][value].weights = np.zeros(num_samples) + (1.0 / num_samples) # Evenly weight each sample for now

	# We don't initialize any values into messages_next
	messages_next[key][value] = Message(num_samples, source_dim, target_dim)