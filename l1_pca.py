import numpy as np

# Using this algorithm.

def generate_matrices(N, K):
	num_mats = 2**(N*K)
	mat_list = []
	for i in range(num_mats):
		bin_i = np.binary_repr(i, width=N*K)
		mat = np.zeros(N*K)
		for j in range(N*K):
			mat[j] = int(bin_i[j])
		mat_list.append(np.reshape(mat, (N,K)))
	return mat_list