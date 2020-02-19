import numpy as np
import matplotlib.pyplot as plt

# landmark_coords = [
# 	np.array([0., 0.]),
# 	np.array([5., 5.]),
# 	np.array([0.,10.])
# ]
# num_landmarks = len(landmark_coords)
landmark_coords = []
num_landmarks = 3
for _ in range(num_landmarks):
	landmark_coords.append(np.random.uniform(low=0.0, high=10.0, size=(2)))

# Build up a list of points to measure at
x_vals = np.linspace(0, 10, num=81)
y_vals = np.linspace(0, 10, num=81)
xx, yy = np.meshgrid(x_vals, y_vals)
points = np.stack((np.ravel(xx), np.ravel(yy)), axis=-1)
num_points = len(points)

range_data = np.zeros((num_points, num_landmarks))

for i in range(num_points):
	for j in range(num_landmarks):
		range_data[i][j] = np.linalg.norm(points[i] - landmark_coords[j])

from sklearn.manifold import LocallyLinearEmbedding, MDS, Isomap, SpectralEmbedding, TSNE

method = Isomap(n_neighbors=8, n_components=2, n_jobs=-1)
feature_coords = method.fit_transform(range_data)

fig, ax = plt.subplots()
ax.scatter(feature_coords[:,0], feature_coords[:,1], c=points[:,0]/10.0, cmap=plt.cm.Spectral)
plt.show()