import numpy as np

def compute_ltsa(points, neighbor_dict, tangents, source_dim, target_dim):
	# This was helpful: https://gitlab.com/charles1992/ltsa/blob/master/ltsa/_local_tangent_space_alignment.py
	num_points = len(points)
	#
	# Compute the Theta_i and Theta_i^+
	Thetas = []
	Thetas_inv = []
	for i in range(num_points):
		neighbors = np.concatenate((neighbor_dict[i], [i]), axis=0)
		k = len(neighbors)
		X_i = points[neighbors]
		X_i_bar = np.dot(X_i.T, (np.ones((k, k)) / k))
		X_i_c = X_i.T - X_i_bar

		Qi = tangents[i].T
		# print "Qi", i, Qi
		Theta_i = np.matmul(Qi.T, X_i_c)
		# print "Theta_i", Theta_i
		Thetas.append(Theta_i)
		Thetas_inv.append(np.linalg.pinv(Theta_i))
	
	# print "len(Thetas)", len(Thetas)
	# print "Thetas[0].shape", Thetas[0].shape
	# print "Thetas", Thetas
	# Compute B from the Wi's
	B = np.zeros((num_points, num_points))
	for i in range(num_points):
		neighbors = np.concatenate((neighbor_dict[i], [i]), axis=0)
		k = len(neighbors)
		Wi_left = np.eye(k) - (1.0/k)
		T_prod = np.matmul(Thetas_inv[i], Thetas[i])
		Wi_right = np.eye(k) - T_prod
		Wi = np.matmul(Wi_left, Wi_right)
		#
		# print "nbd", i, np.ix_(neighbors, neighbors)
		B[np.ix_(neighbors, neighbors)] = B[np.ix_(neighbors, neighbors)] + Wi
		#
	# Compute T
	eig_vals, eig_vecs = np.linalg.eig(B)
	# print eig_vals
	# print eig_vecs
	sort = eig_vals.argsort() # Sort smallest -> largest
	eig_vals.sort()
	eig_vecs = eig_vecs[:, sort]
	T = eig_vecs[:, 1:(target_dim+1)]
	#
	return T

# X = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
# tangents = np.array([[[1, 0]], [[1, 0]], [[1, 0]], [[1, 0]], [[1, 0]]])
# k=2

# def sparseMatrixToDict(mat):
# 	# https://stackoverflow.com/questions/52322847/what-is-an-efficient-way-to-convert-an-adjacency-matrix-to-a-dictionary
# 	return {i: [j for j, adjacent in enumerate(row) if adjacent] for i, row in enumerate(mat.toarray())}

# def sparseMaximum(A, B):
# 	# https://stackoverflow.com/questions/19311353/element-wise-maximum-of-two-sparse-matrices
# 	BisBigger = A-B
# 	BisBigger.data = np.where(BisBigger.data < 0, 1, 0)
# 	return A - A.multiply(BisBigger) + B.multiply(BisBigger)

# from sklearn.neighbors import kneighbors_graph
# neighbor_graph = kneighbors_graph(X, k, mode="distance", n_jobs=-1)
# # neighbor_graph = sparseMaximum(neighbor_graph, neighbor_graph.T)
# neighbor_dict = sparseMatrixToDict(neighbor_graph)
# print compute_ltsa(X, neighbor_dict, tangents, 2, 1)