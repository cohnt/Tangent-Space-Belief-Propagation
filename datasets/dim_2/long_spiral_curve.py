import numpy as np
from scipy.linalg import orth

def make_long_spiral_curve(n_samples, noise_factor, b_val=0.05, rs_seed=None):
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
	# x(t) = btcos(t)
	# y(t) = btsin(t)
	# for 2pi <= t <= 12pi
	# See for graph: https://www.desmos.com/calculator/ete1b4p535
	lowerBound = 2.0 * np.pi
	upperBound = 8.0 * np.pi
	
	# For computing the Jacobian, we have
	# dx/dt = bcos(t) - btsin(t)
	# dy/dt = bsin(t) + btcos(t)

	t = rs.uniform(lowerBound, upperBound, n_samples)
	data = np.array([b_val * t * np.cos(t), b_val * t * np.sin(t)]).transpose()
	# data.shape will be (n_samples, 2), so we can interpret it as a list of n_samples 2D points
	ts = np.array([(b_val * np.cos(t)) - (b_val * t * np.sin(t)), (b_val * np.sin(t)) + (b_val * t * np.cos(t))]).transpose().reshape(n_samples, 1, 2)
	for i in range(n_samples):
		ts[i] = orth(ts[i].T).T

	# Add Gaussian noise to the samples
	mean = [0, 0]
	cov = [[noise_factor, 0], [0, noise_factor]]
	noise = rs.multivariate_normal(mean, cov, n_samples)

	color = (t - lowerBound) / (upperBound - lowerBound)

	return (data + noise, color, ts, rs_seed)

if __name__ == "__main__":
	import matplotlib.pyplot as plt
	data, color, ts = make_long_spiral_curve(500, 0.001, b_val=0.05)
	plt.scatter(data[:,0], data[:,1], c=color, cmap=plt.cm.Spectral)
	plt.show()