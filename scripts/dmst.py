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

t = 4

output_dir = "other_results/results_dmst/"

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

f.write("\n[DMST]\n")
f.write("t=%d\n" % t)

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

###################
# Distance Matrix #
###################
from scipy.spatial import distance_matrix

dist_mat = distance_matrix(points, points)

########
# DMST #
########
from scipy.sparse import csr_matrix
from visualization.plot_neighbors import plot_neighbors_2d

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

edge_list = []
for i in range(0, num_points):
	for j in range(i+1, num_points):
		edge_list.append(
			((i, j), dist_mat[i, j])
		)

sorted_edge_list = sorted(edge_list, key=lambda tup: tup[1])

edge_array = np.zeros((num_points, num_points))

for iter_t in range(t):
	print "Computing DMST %d" % iter_t
	connected_components = DisjointSet(range(num_points))
	edge_idx = 0
	edge_iter_list = []
	while len(connected_components.sets) > 1:
		while True:
			i = sorted_edge_list[edge_idx][0][0]
			j = sorted_edge_list[edge_idx][0][1]
			set_i = connected_components.findSet(i)
			if not j in set_i:
				set_j = connected_components.findSet(j)
				connected_components.union(set_i, set_j)
				edge_array[i, j] = 1
				edge_iter_list.append(edge_idx)
				edge_idx = edge_idx + 1
				break
			else:
				edge_idx = edge_idx + 1
	edge_iter_list.reverse()
	for idx in edge_iter_list:
		sorted_edge_list.pop(idx)
	fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
	plot_neighbors_2d(points, color, csr_matrix(edge_array), ax, point_size=2, line_width=0.25, edge_thickness=0.5, show_labels=False)
	ax.set_title("Improved Neighbors Graph at t=%d" % (iter_t + 1))
	plt.savefig(output_dir + "better_neighbors_%d.svg" % (iter_t + 1))
	plt.close(fig)
