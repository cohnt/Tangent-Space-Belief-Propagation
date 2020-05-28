import numpy as np
from sklearn import preprocessing

# Using this algorithm https://iehttps://ieeexplore-ieee-org.proxy.lib.umich.edu/stamp/stamp.jsp?tp=&arnumber=6629811eexplore-ieee-org.proxy.lib.umich.edu/stamp/stamp.jsp?tp=&arnumber=6629811

def l1_pca(X, K):
	X = preprocessing.scale(X, axis=1)
	D, N = X.shape
	B_candidates = generate_matrices(N, K)
	best_ind = 0
	best_norm_val = np.linalg.norm(np.matmul(X, B_candidates[0]), ord="nuc")
	for i in range(1, len(B_candidates)):
		norm_val = np.linalg.norm(np.matmul(X, B_candidates[i]), ord="nuc")
		if norm_val > best_norm_val:
			best_norm_val = norm_val
			best_ind = i
	B_opt = B_candidates[best_ind]
	U, S, VT = np.linalg.svd(np.matmul(X, B_opt), full_matrices=True)
	V = VT.T
	Rl1 = np.matmul(U[:,0:K], V.T)
	return Rl1

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