import numpy as np
import matplotlib.pyplot as plt

def relativeErrorBarChart(ax, dict):
	rect = ax.bar(range(len(dict)), list(dict.values()), align='center')
	ax.set_title("Reconstruction Error by Manifold Learning Algorithm")
	ax.set_ylabel("Error")
	ax.set_xticks(range(len(dict)))
	ax.set_xticklabels(list(dict.keys()))
	ax.tick_params(axis='x', which='major', labelsize=10)

	autolabel(ax, rect)

def autolabel(ax, rects, xpos='center'):
	xpos = xpos.lower()
	ha = {'center': 'center', 'right': 'left', 'left': 'right'}
	offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

	for rect in rects:
		height = rect.get_height()
		ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
		        '{0:.4f}'.format(height), ha=ha[xpos], va='bottom', size=10)

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
def regressionErrorCharacteristic(ax, embedded_points, true_parameters, dist_metric="l2", density=0.001):
	embedded_dists = pairwise_distances(normalize(embedded_points, axis=0, copy=True), metric=dist_metric, n_jobs=-1)
	true_dists = pairwise_distances(normalize(true_parameters, axis=0, copy=True), metric=dist_metric, n_jobs=-1)
	err_mat = np.abs(embedded_dists - true_dists)

	num_points = err_mat.shape[0]
	x_vals = np.sort(err_mat[np.triu_indices(num_points)])
	x_vals = x_vals[np.asarray(np.linspace(0, len(x_vals)-1, num=density*len(x_vals)), dtype=int)]
	y_vals = np.arange(1, len(x_vals)+1) / float(len(x_vals))
	ax.plot(x_vals, y_vals)
	ax.set_xlim(left=0, right=np.max(err_mat))
	ax.set_ylim(bottom=0, top=1)
	ax.set_xlabel("Distance Error Threshold")
	ax.set_ylabel("Proportion of Pairwise Distances")