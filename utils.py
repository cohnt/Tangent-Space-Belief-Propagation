import numpy as np
import sys

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

def sphereRand(pos, radius, variance=0.01):
	# See method 19 from http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
	u = np.random.normal(0, 1, len(pos))
	d = np.sum(u ** 2) ** (0.5)
	D = np.random.normal(d, variance)
	return pos + (radius * u / D)