import numpy as np

def make_s_curve(n_samples, noise_factor):
	"""
		n_samples: number of points to generate
		noise_factor: variance of the noise added to each dimension
		
		For best results, noise_factor should be pretty small (at most 0.01)
	"""

	# The s curve is parameterized by 
	# x(t) = 0.5 + sin(t)cos(t)
	# y(t) = 0.5 + 0.5cos(t)
	# for (3/4)pi <= t <= (9/4)pi
	lowerBound = 3.0 * np.pi / 4.0
	upperBound = 9.0 * np.pi / 4.0
	
	t = np.random.uniform(lowerBound, upperBound, n_samples)
	data = np.array([0.5 + np.multiply(np.sin(t), np.cos(t)), 0.5 + (0.5 * np.cos(t))]).transpose()
	# data.shape will be (n_samples, 2), so we can interpret it as a list of n_samples 2D points

	# Add Gaussian noise to the samples
	mean = [0, 0]
	cov = [[noise_factor, 0], [0, noise_factor]]
	noise = np.random.multivariate_normal(mean, cov, n_samples)

	return data + noise

if __name__ == "__main__":
	import matplotlib.pyplot as plt
	data = make_s_curve(500, 0.001)
	plt.scatter(data[:,0], data[:,1])
	plt.show()