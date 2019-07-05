import numpy as np
from scipy.linalg import orth

def make_s_curve(n_samples, noise_factor, rs_seed=None):
	"""
		n_samples: number of points to generate
		noise_factor: variance of the noise added to each dimension
		
		For best results, noise_factor should be pretty small (at most 0.001)
	"""

	if rs_seed == None:
		rs_seed = np.random.randint(0, 2**32)
	print "Using dataset seed=%d" % rs_seed
	rs = np.random.RandomState(seed=rs_seed)

	# The s curve is parameterized by 
	# x(t) = 0.5 + sin(t)cos(t)
	# y(t) = 0.5 + 0.5cos(t)
	# for (3/4)pi <= t <= (9/4)pi
	# See for graph: https://www.desmos.com/calculator/ete1b4p535
	lowerBound = 3.0 * np.pi / 4.0
	upperBound = 9.0 * np.pi / 4.0
	
	# For computing the Jacobian, we have
	# dx/dt = cos(t)cos(t) - sin(t)sin(t)
	# dy/dt = -0.5sin(t)

	t = rs.uniform(lowerBound, upperBound, n_samples)
	data = np.array([0.5 + np.multiply(np.sin(t), np.cos(t)), 0.5 + (0.5 * np.cos(t))]).transpose()
	# data.shape will be (n_samples, 2), so we can interpret it as a list of n_samples 2D points
	ts = np.array([(np.cos(t) ** 2) - (np.sin(t) ** 2), -0.5 * np.sin(t)]).transpose().reshape(n_samples, 1, 2)
	for i in range(n_samples):
		ts[i] = orth(ts[i].T).T

	# Add Gaussian noise to the samples
	mean = [0, 0]
	cov = [[noise_factor, 0], [0, noise_factor]]
	noise = rs.multivariate_normal(mean, cov, n_samples)

	color = (t - lowerBound) / (upperBound - lowerBound)

	return (data + noise, color, ts)

if __name__ == "__main__":
	import matplotlib.pyplot as plt
	data, color, ts = make_s_curve(500, 0.001)
	plt.scatter(data[:,0], data[:,1], c=color, cmap=plt.cm.Spectral)
	plt.show()