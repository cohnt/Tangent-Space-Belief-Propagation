import numpy as np
import sys
from scipy.stats import special_ortho_group
import matplotlib.pyplot as plt

class Message:
	def __init__(self, num_samples, target_dim):
		self.pos = np.zeros((num_samples, target_dim))
		self.weights = np.zeros(num_samples)

class Belief:
	def __init__(self, num_samples, target_dim):
		self.pos = np.zeros((num_samples, target_dim))
		self.weights = np.zeros(num_samples)

def sparseMatrixToDict(mat):
	# https://stackoverflow.com/questions/52322847/what-is-an-efficient-way-to-convert-an-adjacency-matrix-to-a-dictionary
	return {i: [j for j, adjacent in enumerate(row) if adjacent] for i, row in enumerate(mat.toarray())}

def sparseMaximum(A, B):
	# https://stackoverflow.com/questions/19311353/element-wise-maximum-of-two-sparse-matrices
	BisBigger = A-B
	BisBigger.data = np.where(BisBigger.data < 0, 1, 0)
	return A - A.multiply(BisBigger) + B.multiply(BisBigger)

def write(*args, **kwargs):
	sys.stdout.write(*args, **kwargs)

def flush(*args, **kwargs):
	sys.stdout.flush(*args, **kwargs)

def weightedSample(vec, num):
	# Given a (1d) vector of weights, return num weighted samples (as indices)
	# The weights don't need to be normalized, but it cannot be zero everywhere.
	if vec.dtype != float:
		vec = vec.astype(float)
	return np.random.choice(len(vec), num, p=(vec / sum(vec)))

def list_mvn(means, covs, single_cov=False):
	# Generate multivariate normal random vectors for a list of means and corresponding covariance matrices
	# For n dimension, with m means, means.shape === (m, n) and covs.shape === (m, n, n)
	# The output will be shape (m, n)
	m = means.shape[0]
	n = means.shape[1]
	out = np.zeros((m, n))
	for i in range(0, m):
		if single_cov:
			out[i] = np.random.multivariate_normal(means[i], covs)
		else:
			out[i] = np.random.multivariate_normal(means[i], covs[i])
	return out

def sphereRand(pos, radius, var=0.01):
	# See method 19 from http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
	u = np.random.normal(0, 1, len(pos))
	d = np.sum(u ** 2) ** (0.5)
	D = np.random.normal(d, var)
	return pos + (radius * u / D)

def projSubspace(orthonormal_basis, point):
	individual_components = np.zeros(len(orthonormal_basis))
	for i in range(len(orthonormal_basis)):
		basis_vec = orthonormal_basis[i]
		individual_components[i] = np.dot(basis_vec, point)
	return individual_components

def randomSmallRotation(dimension, variance=None):
	if variance is None:
		variance = 0.05 * dimension * 180.0 / np.pi
	theta = np.random.normal(0, variance) * np.pi / 180.0
	rotMat = np.eye(dimension)
	rotMat[0,0] = np.cos(theta)
	rotMat[0,1] = -np.sin(theta)
	rotMat[1,0] = np.sin(theta)
	rotMat[1,1] = np.cos(theta)
	basis = special_ortho_group.rvs(dimension)
	basis_inv = basis.transpose()
	return basis.dot(rotMat).dot(basis_inv)

def increaseDimensionMatrix(old_dimension, new_dimension):
	expand_matrix = np.zeros((old_dimension, new_dimension))
	expand_matrix[0:old_dimension, 0:old_dimension] = np.eye(old_dimension)
	rotation_matrix = special_ortho_group.rvs(new_dimension)
	final_matrix = np.matmul(expand_matrix, rotation_matrix)
	return final_matrix

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
def pairwiseDistErr(embedded_points, true_parameters, normalize_data=True, normalize_dists=False, dist_metric="l2", mat_norm="fro"):
	embedded_dists = None
	true_dists = None
	if normalize_data:
		embedded_dists = pairwise_distances(normalize(embedded_points, axis=0, copy=True), metric=dist_metric, n_jobs=-1)
		true_dists = pairwise_distances(normalize(true_parameters, axis=0, copy=True), metric=dist_metric, n_jobs=-1)
	else:
		embedded_dists = pairwise_distances(embedded_points, metric=dist_metric, n_jobs=-1)
		true_dists = pairwise_distances(true_parameters, metric=dist_metric, n_jobs=-1)
	err = None
	if mat_norm == "max":
		return np.max(np.abs(embedded_dists - true_dists))
	elif mat_norm == "mean":
		return np.mean(np.abs(embedded_dists - true_dists))
	elif mat_norm == "median":
		return np.median(np.abs(embedded_dists - true_dists))
	else:
		#
		if normalize_dists:
			err = np.linalg.norm((embedded_dists/np.max(embedded_dists)) - (true_dists/np.max(true_dists)), ord=mat_norm, axis=None)
		else:
			err = np.linalg.norm(embedded_dists - true_dists, ord=mat_norm, axis=None)
		return err

def setAxisTickSize(ax, size, n_ticks=None):
	if not n_ticks is None:
		ax.xaxis.set_major_locator(plt.MaxNLocator(n_ticks))
		ax.yaxis.set_major_locator(plt.MaxNLocator(n_ticks))
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(size)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(size)

def setAxisTickSize3D(ax, size, n_ticks=None):
	if not n_ticks is None:
		ax.xaxis.set_major_locator(plt.MaxNLocator(n_ticks))
		ax.yaxis.set_major_locator(plt.MaxNLocator(n_ticks))
		ax.zaxis.set_major_locator(plt.MaxNLocator(n_ticks))
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(size)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(size)
	for tick in ax.zaxis.get_major_ticks():
		tick.label.set_fontsize(size)	