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

dataset_name = "long_spiral_curve"
# dataset_seed = 4045775215
dataset_seed = np.random.randint(0, 2**32)
num_points = 500    # Number of data points
data_noise = 0.001 # How much noise is added to the data
source_dim = 2      # The dimensionality of the incoming dataset (see "Load Dataset" below)
target_dim = 1      # The number of dimensions the data is being reduced to

num_iters = 25     # Number of iterations of the message passing algorithm to run
neighbors_k = 12    # The value of 'k' used for k-nearest-neighbors
num_samples = 10   # Numbers of samples used in the belief propagation algorithm
explore_perc = 0.1  # Fraction of uniform samples to keep exploring

message_resample_cov = np.eye(target_dim) * 0.01 # TODO: Change
pruning_angle_thresh = np.cos(30.0 * np.pi / 180.0)

output_dir = "other_results/results_tsne/"
error_histogram_num_bins = num_points / 10
err_dist_metric = "l2"
err_mat_norm = "fro"

embedding_name = "KernelPCA" # Could also be MDS
kpca_eigen_solver = "auto"
kpca_tol = 1e-9
kpca_max_iter = 3000

data_sp_rad = 7.0
data_sp_lw = 1.0
nn_lw = 1.0
pca_ll = 0.1
embedding_sp_rad = 7.0
embedding_sp_lw = 1.0
combined_sp_rad = 4.0
combined_sp_lw = 0.5

write("\n")

matplotlib.rcParams.update({'font.size': 15})

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
ax.scatter(points[:,0], points[:,1], c=color, cmap=plt.cm.Spectral, s=data_sp_rad**2, zorder=2, linewidth=data_sp_lw)
ax.set_title("Dataset (num=%d, variance=%f, seed=%d)\n" % (num_points, data_noise, dataset_seed))
plt.savefig(output_dir + "dataset.svg")
plt.close(fig)
t1 = time.time()
write("Done! dt=%f\n" % (t1-t0))
flush()

#########################
# Test t-SNE parameters #
#########################

from sklearn.manifold import TSNE

perplexity_range = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
learning_rate_range = np.array([10, 15, 20, 25, 50, 100, 150, 200, 250, 500, 750, 1000], dtype=float)

for perplexity in perplexity_range:
	for learning_rate in learning_rate_range:
		for test_num in range(0, 10):
			tsne = TSNE(n_components=target_dim, perplexity=perplexity, learning_rate=learning_rate, init="random")

			write("Computing perplexity %.1f\tlearning_rate %.1f\ttest_num %d..." % (perplexity, learning_rate, test_num))
			flush()
			t0 = time.time()
			feature_coords = tsne.fit_transform(points)
			t1 = time.time()
			write("Done! dt=%f\n" % (t1-t0))
			flush()

			err = pairwiseDistErr(feature_coords, true_parameters, dist_metric=err_dist_metric, mat_norm=err_mat_norm)
			print "Embedding error: %f" % err

			fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
			ax.scatter(color, feature_coords, c=color, cmap=plt.cm.Spectral, s=embedding_sp_rad**2, linewidths=embedding_sp_lw)
			ax.set_title("perplexity: %.1f learning_rate: %.1f test_num: %d\nEmbedding error: %f" % (perplexity, learning_rate, test_num, err))
			plt.xlabel("Actual Parameter Value")
			plt.ylabel("Embedded Coordinate")
			plt.savefig(output_dir + ("tsne_%.1f_%.1f_%d.svg" % (perplexity, learning_rate, test_num)))
			plt.close(fig)

		tsne = TSNE(n_components=target_dim, perplexity=perplexity, learning_rate=learning_rate, init="pca")

		write("Computing perplexity %.1f\tlearning_rate %.1f\tPCA init..." % (perplexity, learning_rate))
		flush()
		t0 = time.time()
		feature_coords = tsne.fit_transform(points)
		t1 = time.time()
		write("Done! dt=%f\n" % (t1-t0))
		flush()

		err = pairwiseDistErr(feature_coords, true_parameters, dist_metric=err_dist_metric, mat_norm=err_mat_norm)
		print "Embedding error: %f" % err

		fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
		ax.scatter(color, feature_coords, c=color, cmap=plt.cm.Spectral, s=embedding_sp_rad**2, linewidths=embedding_sp_lw)
		ax.set_title("perplexity: %.1f learning_rate: %.1f (PCA init)\nEmbedding error: %f" % (perplexity, learning_rate, err))
		plt.xlabel("Actual Parameter Value")
		plt.ylabel("Embedded Coordinate")
		plt.savefig(output_dir + ("tsne_%.1f_%.1f_PCA.svg" % (perplexity, learning_rate)))
		plt.close(fig)