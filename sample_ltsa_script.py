import numpy as np
from sklearn.neighbors import kneighbors_graph

X = np.array([[0, 0], [1, 0], [2, 0.25], [3, 0.5], [4, 1]])

k = 2
neighbor_graph = kneighbors_graph(X, k, mode="distance", n_jobs=-1).toarray()
neighbor_graph = np.maximum(neighbor_graph, neighbor_graph.T)

B = np.zeros((len(X), len(X)))
for i in range(len(X)):
    print ""
    print "---------------------"
    print ""
    nbd = np.nonzero(neighbor_graph[i])[0]
    print "nbd", nbd
    Xi = (X[nbd]).T
    print "Xi", Xi
    x_bar = np.average(Xi, axis=1).reshape(2, 1)
    print "x_bar", x_bar
    e = np.ones((len(Xi[0]), 1))
    print "e", e
    print "e.shape", e.shape
    half_mat = Xi - np.matmul(x_bar, e.T)
    print "half_mat", half_mat
    cor_mat = np.matmul(half_mat.T, half_mat)
    print "cor_mat", cor_mat
    print ""
    w, v = np.linalg.eig(cor_mat)
    print "w", w
    print "v", v
    v = v.T
    v = v[w.argsort()]
    v = v.T
    w.sort()
    print "sorted w", w
    print "sorted v", v
    d = 1
    Gi = np.zeros((len(v[0]), d+1))
    Gi[:,0] = 1.0/np.sqrt(2)
    Gi[:,1] = v[:,-1]
    print "Gi", Gi
    B[np.ix_(nbd, nbd)] = B[np.ix_(nbd, nbd)] + np.eye(len(Gi)) - np.matmul(Gi, Gi.T)

print ""
print "==================="
print ""
print "B", B

w, v = np.linalg.eig(B)
print "w", w
print "v", v
v = v.T
v = v[w.argsort()]
v = v.T
w.sort()
print "sorted w", w
print "sorted v", v
Tt = np.zeros((len(v[0]), d))
Tt[:,0] = v[:,1]
T = Tt.T
print "T", T
