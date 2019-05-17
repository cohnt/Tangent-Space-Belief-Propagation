import numpy as np

def make_o_curve(n_samples, noise_factor):
	"""
		n_samples: number of points to generate
		noise_factor: variance of the noise added to each dimension
		
		For best results, noise_factor should be pretty small (at most 0.001)
	"""

	# The o curve (inspired my the LaTeX rendering of \mathcal{O}) is parameterized by 
	# x(t) = 0.5 + 0.5sin(t)
	# y(t) = 0.5 + 0.5cos(t)+0.02t
	# for (-1/4)pi <= t <= (9/4)pi
	lowerBound = -1.0 * np.pi / 4.0
	upperBound = 9.0 * np.pi / 4.0
	
	t = np.random.uniform(lowerBound, upperBound, n_samples)
	data = np.array([0.5 + (0.5 * np.sin(t)), 0.5 + (0.5 * np.cos(t)) + (0.02 * t)]).transpose()
	# data.shape will be (n_samples, 2), so we can interpret it as a list of n_samples 2D points

	# Add Gaussian noise to the samples
	mean = [0, 0]
	cov = [[noise_factor, 0], [0, noise_factor]]
	noise = np.random.multivariate_normal(mean, cov, n_samples)

	return data + noise

if __name__ == "__main__":
	import matplotlib.pyplot as plt
	data = make_o_curve(500, 0.0005)
	plt.scatter(data[:,0], data[:,1])
	plt.show()