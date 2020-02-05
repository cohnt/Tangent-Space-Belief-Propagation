import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from textwrap import wrap

from autoencoder import Autoencoder

############
# 2D -> 1D #
############

dataset_name = "long_spiral_curve"
dataset_seed = 4045775215 # np.random.randint(0, 2**32)
num_points = 5000 # Number of data points
data_noise = 0.001 # How much noise is added to the data
source_dim = 2 # The dimensionality of the incoming dataset (see "Load Dataset" below)
target_dim = 1 # The number of dimensions the data is being reduced to

data_sp_rad = 7.0
data_sp_lw = 1.0
embedding_sp_rad = 7.0
embedding_sp_lw = 1.0

from datasets.dim_2.arc_curve import make_arc_curve
from datasets.dim_2.s_curve import make_s_curve
from datasets.dim_2.o_curve import make_o_curve
from datasets.dim_2.eight_curve import make_eight_curve
from datasets.dim_2.long_spiral_curve import make_long_spiral_curve
from sklearn.preprocessing import MinMaxScaler

points, color, true_tangents, true_parameters, dataset_seed = make_s_curve(num_points, data_noise, rs_seed=dataset_seed)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(points[:,0], points[:,1], c=color, cmap=plt.cm.Spectral, s=data_sp_rad**2, zorder=2, linewidth=data_sp_lw)
ax.set_title("Dataset (num=%d, variance=%f, seed=%d)\n" % (num_points, data_noise, dataset_seed))
plt.show()

############################

autoencoder = Autoencoder(2, 1, [64, 32, 32], ["relu", "relu", "relu"])
autoencoder.train(points)
embedded_points = autoencoder.embed(points)
reconstructed_points = autoencoder.reconstruct(points)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(reconstructed_points[:,0], reconstructed_points[:,1], c=color, cmap=plt.cm.Spectral, s=data_sp_rad**2, zorder=2, linewidth=data_sp_lw)
ax.set_title("Reconstructed Data")
plt.show()

############################

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(color, embedded_points, c=color, cmap=plt.cm.Spectral, s=embedding_sp_rad**2, linewidths=embedding_sp_lw)
ax.set_title("\n".join(wrap("Actual Parameter Value vs Embedded Coordinate from Autoencoder", 50)))
plt.xlabel("Actual Parameter Value")
plt.ylabel("Embedded Coordinate")
plt.show()

############
# 3D -> 1D #
############

from mpl_toolkits.mplot3d import Axes3D

dataset_name = "tight_spiral_curve"
dataset_seed = np.random.randint(0, 2**32)
num_points = 5000 # Number of data points
data_noise = 0.0005 # How much noise is added to the data
source_dim = 3 # The dimensionality of the incoming dataset (see "Load Dataset" below)
target_dim = 1 # The number of dimensions the data is being reduced to

from datasets.dim_3.helix_curve import make_helix_curve
from datasets.dim_3.tight_spiral_curve import make_tight_spiral_curve

points, color, true_tangents, dataset_seed = make_tight_spiral_curve(num_points, data_noise, rs_seed=dataset_seed)
fig = plt.figure(figsize=(14.4, 10.8), dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2], c=color, cmap=plt.cm.Spectral, s=data_sp_rad**2, linewidth=data_sp_lw)
ax.set_title("Dataset (num=%d, variance=%f, seed=%d)" % (num_points, data_noise, dataset_seed))
plt.show()

############################

autoencoder = Autoencoder(3, 1, [64, 32, 32], ["relu", "relu", "relu"])
autoencoder.train(points)
embedded_points = autoencoder.embed(points)
reconstructed_points = autoencoder.reconstruct(points)

fig = plt.figure(figsize=(14.4, 10.8), dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reconstructed_points[:,0], reconstructed_points[:,1], reconstructed_points[:,2], c=color, cmap=plt.cm.Spectral, s=data_sp_rad**2, linewidth=data_sp_lw)
ax.set_title("Reconstructed Dataset")
plt.show()

############################

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(color, embedded_points, c=color, cmap=plt.cm.Spectral, s=embedding_sp_rad**2, linewidths=embedding_sp_lw)
ax.set_title("\n".join(wrap("Actual Parameter Value vs Embedded Coordinate from Autoencoder", 50)))
plt.xlabel("Actual Parameter Value")
plt.ylabel("Embedded Coordinate")
plt.show()

############
# 3D -> 2D #
############

dataset_name = "swiss_roll"
dataset_seed = np.random.randint(0, 2**32)
num_points = 350    # Number of data points
data_noise = 0.001     # How much noise is added to the data
source_dim = 3      # The dimensionality of the incoming dataset (see "Load Dataset" below)
target_dim = 2      # The number of dimensions the data is being reduced to

from datasets.dim_3.s_sheet import make_s_sheet
from datasets.dim_3.swiss_roll import make_swiss_roll_sheet

disp_elev = 5.0
disp_azim = -85.0

def make3DFigure():
	f = plt.figure(figsize=(14.4, 10.8), dpi=100)
	a = f.add_subplot(111, projection='3d')
	if dataset_name == "swiss_roll":
		a.set_ylim(bottom=-0.5, top=1.5)
	a.view_init(elev=disp_elev, azim=disp_azim)
	return f, a

points, color, true_tangents, dataset_seed = make_swiss_roll_sheet(num_points, data_noise, rs_seed=dataset_seed)
fig, ax = make3DFigure()
ax.scatter(points[:,0], points[:,1], points[:,2], c=color, cmap=plt.cm.Spectral, s=data_sp_rad**2, linewidth=data_sp_lw)
ax.set_title("Dataset (num=%d, variance=%f, seed=%d)" % (num_points, data_noise, dataset_seed))
plt.show()

############################

autoencoder = Autoencoder(3, 2, [64, 32, 32], ["relu", "relu", "relu"])
autoencoder.train(points)
embedded_points = autoencoder.embed(points)

############################

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(embedded_points[:,0], embedded_points[:,1], c=color, cmap=plt.cm.Spectral, s=embedding_sp_rad**2, linewidths=embedding_sp_lw)
ax.set_title("\n".join(wrap("2D Embedding from Autoencoder", 60)))
plt.show()