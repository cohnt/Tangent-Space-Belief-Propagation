import numpy as np
import sys

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
	if vec.dtype != float:
		vec = vec.astype(float)
	return np.random.choice(len(vec), num, p=(vec / sum(vec)))