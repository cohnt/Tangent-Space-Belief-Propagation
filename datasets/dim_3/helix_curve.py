import numpy as np

def make_helix_curve(n_samples, noise_factor):
	"""
		n_samples: number of points to generate
		noise_factor: variance of the noise added to each dimension
		
		For best results, noise_factor should be pretty small (at most 0.001)
	"""

	# The helix curve is parameterized by 
	# x(t) = cos(t)
	# y(t) = sin(t)
	# z(t) = t/4pi
	# for 0 <= t <= 4pi
	lowerBound = 0.0
	upperBound = 4.0 * np.pi
	
	# For computing the Jacobian, we have
	# dx/dt = -sin(t)
	# dy/dt = cos(t)
	# dz/dt = 1/4pi

	t = np.random.uniform(lowerBound, upperBound, n_samples)
	data = np.array([np.cos(t), np.sin(t), t / (4*np.pi)]).transpose()
	# data.shape will be (n_samples, 3), so we can interpret it as a list of n_samples 3D points
	ts = np.array([-np.sin(t), np.cos(t), np.full(t.shape, 1 / (4*np.pi))]).transpose().reshape(n_samples, 1, 3)
	ts_norm = np.apply_along_axis(np.linalg.norm, 2, ts).reshape(n_samples, 1, 1)
	ts = ts / ts_norm

	# Add Gaussian noise to the samples
	mean = [0, 0, 0]
	cov = [[noise_factor, 0, 0],
	       [0, noise_factor, 0],
	       [0, 0, noise_factor]]
	noise = np.random.multivariate_normal(mean, cov, n_samples)

	color = (t - lowerBound) / (upperBound - lowerBound)

	return (data + noise, color, ts)

if __name__ == "__main__":
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.pyplot as plt
	data, color, ts = make_helix_curve(500, 0.001)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(data[:,0], data[:,1], data[:,2], c=color, cmap=plt.cm.Spectral)
	plt.show()