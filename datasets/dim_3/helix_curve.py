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
	
	t = np.random.uniform(lowerBound, upperBound, n_samples)
	data = np.array([np.cos(t), np.sin(t), t / (4*np.pi)]).transpose()
	# data.shape will be (n_samples, 3), so we can interpret it as a list of n_samples 3D points

	# Add Gaussian noise to the samples
	mean = [0, 0, 0]
	cov = [[noise_factor, 0, 0],
	       [0, noise_factor, 0],
	       [0, 0, noise_factor]]
	noise = np.random.multivariate_normal(mean, cov, n_samples)

	color = (t - lowerBound) / (upperBound - lowerBound)

	return (data + noise, color)

if __name__ == "__main__":
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.pyplot as plt
	data, color = make_helix_curve(500, 0.001)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(data[:,0], data[:,1], data[:,2], c=color, cmap=plt.cm.Spectral)
	plt.show()