import numpy as np
from scipy.linalg import orth

def make_tight_spiral_curve(n_samples, noise_factor, rs_seed=None):
	"""
		n_samples: number of points to generate
		noise_factor: variance of the noise added to each dimension
		
		For best results, noise_factor should be pretty small (at most 0.001)
	"""

	if rs_seed == None:
		rs_seed = np.random.randint(0, 2**32)
	print "Using dataset seed=%d" % rs_seed
	rs = np.random.RandomState(seed=rs_seed)

	# The helix curve is parameterized by 
	# x(t) = cos(t)
	# y(t) = sin(t)
	# z(t) = t/8pi
	# for 0 <= t <= 8pi
	lowerBound = 0.0
	upperBound = 8.0 * np.pi
	
	# For computing the Jacobian, we have
	# dx/dt = -sin(t)
	# dy/dt = cos(t)
	# dz/dt = 1/4pi

	t = rs.uniform(lowerBound, upperBound, n_samples)
	data = np.array([np.cos(t), np.sin(t), t / (8*np.pi)]).transpose()
	# data.shape will be (n_samples, 3), so we can interpret it as a list of n_samples 3D points
	ts = np.array([-np.sin(t), np.cos(t), np.full(t.shape, 1 / (8*np.pi))]).transpose().reshape(n_samples, 1, 3)
	for i in range(n_samples):
		ts[i] = orth(ts[i].T).T

	# Add Gaussian noise to the samples
	mean = [0, 0, 0]
	cov = [[noise_factor, 0, 0],
	       [0, noise_factor, 0],
	       [0, 0, noise_factor]]
	noise = rs.multivariate_normal(mean, cov, n_samples)

	color = (t - lowerBound) / (upperBound - lowerBound)

	return (data + noise, color, ts, rs_seed)

if __name__ == "__main__":
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.pyplot as plt
	data, color, ts, seed = make_tight_spiral_curve(500, 0.001)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(data[:,0], data[:,1], data[:,2], c=color, cmap=plt.cm.Spectral)
	plt.show()