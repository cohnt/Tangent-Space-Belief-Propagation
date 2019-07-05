import numpy as np
from scipy.linalg import orth

def make_s_sheet(n_samples, noise_factor, rs_seed=None):
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
	# x(s,t) = 0.5 + sin(t)cos(t)
	# y(s,t) = 0.5 + 0.5cos(t)
	# z(s,t) = s
	# for 0 <= s <= 1 and (3/4)pi <= t <= (9/4)pi
	sLowerBound = 0.0
	sUpperBound = 1.0
	tLowerBound = 3.0 * np.pi / 4.0
	tUpperBound = 9.0 * np.pi / 4.0

	# For computing the Jacobian, we have
	# dx/dt = cos(t)cos(t) - sin(t)sin(t)
	# dy/dt = -0.5sin(t)
	# dz/dt = 0
	# 
	# dx/ds = 0
	# dy/ds = 0
	# dz/ds = 1

	s = rs.uniform(sLowerBound, sUpperBound, n_samples)
	t = rs.uniform(tLowerBound, tUpperBound, n_samples)
	data = np.array([0.5 + np.multiply(np.sin(t), np.cos(t)), 0.5 + (0.5 * np.cos(t)), s]).transpose()
	# data.shape will be (n_samples, 3), so we can interpret it as a list of n_samples 3D points
	ts = np.array([[(np.cos(t) ** 2) - (np.sin(t) ** 2), -0.5 * np.sin(t), np.full(t.shape, 0)], [np.full(t.shape, 0), np.full(t.shape, 0), np.full(t.shape, 1)]])
	# ts is currently of shape (2, 3, n_samples)
	ts = ts.transpose()
	# ts is currently of shape (n_samples, 3, 2)
	ts = np.swapaxes(ts, 1, 2)
	# ts is finally of shape (n_samples, 2, 3)
	for i in range(n_samples):
		ts[i] = orth(ts[i].T).T

	# Add Gaussian noise to the samples
	mean = [0, 0, 0]
	cov = [[noise_factor, 0, 0],
	       [0, noise_factor, 0],
	       [0, 0, noise_factor]]
	noise = rs.multivariate_normal(mean, cov, n_samples)

	color = (t - tLowerBound) / (tUpperBound - tLowerBound)

	return (data + noise, color, ts)

if __name__ == "__main__":
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.pyplot as plt
	data, color, ts = make_s_sheet(500, 0.001)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(data[:,0], data[:,1], data[:,2], c=color, cmap=plt.cm.Spectral)
	plt.show()