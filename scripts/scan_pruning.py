import numpy as np
import scipy
import random
import matplotlib
from textwrap import wrap
import matplotlib.pyplot as plt
import time
import sys
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import write, flush

from datasets.other.laser_scan import make_laser_scan_curve

points, true_vals = make_laser_scan_curve()

from sklearn.manifold import LocallyLinearEmbedding, MDS, Isomap, SpectralEmbedding
methods = []
methods.append(LocallyLinearEmbedding(n_neighbors=3, n_components=1, n_jobs=-1))
methods.append(MDS(n_components=1, n_jobs=-1))
methods.append(Isomap(n_neighbors=3, n_components=1, n_jobs=-1))
methods.append(SpectralEmbedding(n_components=1, n_neighbors=3, n_jobs=-1))
num_methods = len(methods)

method_names = ["LLE", "MDS", "Isomap", "SpectralEmbedding"]

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
	
	fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
	ax.scatter(true_vals, feature_coords)
	ax.set_title("\n".join(wrap("Actual Parameter Value vs Embedded Coordinate from %s" % name, 60)))
	plt.xlabel("Actual Parameter Value")
	plt.ylabel("Embedded Coordinate")
	plt.show()