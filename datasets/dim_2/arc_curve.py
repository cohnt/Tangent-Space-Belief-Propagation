import numpy as np

def make_arc_curve(n_samples, noise_factor):
	"""
		n_samples: number of points to generate
		noise_factor: variance of the noise added to each dimension
		
		For best results, noise_factor should be pretty small (at most 0.001)
	"""

	# The arc curve is parameterized by 
	# x(t) = 0.5 + 0.5cos(t)
	# y(t) = 0.5sin(t)
	# for 0 <= t <= pi
	# See for graph: https://www.desmos.com/calculator/ete1b4p535
	lowerBound = 0.0
	upperBound = np.pi
	
	t = np.random.uniform(lowerBound, upperBound, n_samples)
	data = np.array([0.5 + (0.5 * np.cos(t)), 0.5 * np.sin(t)]).transpose()
	# data.shape will be (n_samples, 2), so we can interpret it as a list of n_samples 2D points

	# Add Gaussian noise to the samples
	mean = [0, 0]
	cov = [[noise_factor, 0], [0, noise_factor]]
	noise = np.random.multivariate_normal(mean, cov, n_samples)

	return data + noise

if __name__ == "__main__":
	import matplotlib.pyplot as plt
	data = make_arc_curve(500, 0.001)
	plt.scatter(data[:,0], data[:,1])
	plt.show()