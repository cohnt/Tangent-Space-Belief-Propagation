import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from textwrap import wrap

############
# 2D -> 1D #
############

dataset_name = "long_spiral_curve"
dataset_seed = 4045775215 # np.random.randint(0, 2**32)
num_points = 500 # Number of data points
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

points, color, true_tangents, dataset_seed = make_long_spiral_curve(num_points, data_noise, rs_seed=dataset_seed)

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(points[:,0], points[:,1], c=color, cmap=plt.cm.Spectral, s=data_sp_rad**2, zorder=2, linewidth=data_sp_lw)
ax.set_title("Dataset (num=%d, variance=%f, seed=%d)\n" % (num_points, data_noise, dataset_seed))
plt.show()

############################

from keras.layers import Input, Dense
from keras.models import Model

input_layer = Input(shape=(source_dim,))
encoder_layer_1 = Dense(64, activation="tanh")(input_layer)
encoder_layer_2 = Dense(32, activation="tanh")(encoder_layer_1)
encoder_layer_3 = Dense(32, activation="tanh")(encoder_layer_2)
choke_layer = Dense(target_dim, activation="tanh")(encoder_layer_3)
decoder_layer_3 = Dense(32, activation="tanh")(choke_layer)
decoder_layer_2 = Dense(32, activation="tanh")(decoder_layer_3)
decoder_layer_1 = Dense(64, activation="tanh")(decoder_layer_2)
output_layer = Dense(source_dim, activation="linear")(decoder_layer_1)

encoder = Model(input_layer, choke_layer)
autoencoder = Model(input_layer, output_layer)

train_test_cutoff = int(0.8 * num_points)

autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
autoencoder.fit(
	points[:train_test_cutoff], points[:train_test_cutoff],
	epochs=50,
	batch_size=256,
	shuffle=True,
	validation_data=(points[train_test_cutoff:], points[train_test_cutoff:])
)

embedded_points = encoder.predict(points)

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
num_points = 500 # Number of data points
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

input_layer = Input(shape=(source_dim,))
encoder_layer_1 = Dense(64, activation="tanh")(input_layer)
encoder_layer_2 = Dense(32, activation="tanh")(encoder_layer_1)
encoder_layer_3 = Dense(32, activation="tanh")(encoder_layer_2)
choke_layer = Dense(target_dim, activation="tanh")(encoder_layer_3)
decoder_layer_3 = Dense(32, activation="tanh")(choke_layer)
decoder_layer_2 = Dense(32, activation="tanh")(decoder_layer_3)
decoder_layer_1 = Dense(64, activation="tanh")(decoder_layer_2)
output_layer = Dense(source_dim, activation="linear")(decoder_layer_1)

encoder = Model(input_layer, choke_layer)
autoencoder = Model(input_layer, output_layer)

train_test_cutoff = int(0.8 * num_points)

autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
autoencoder.fit(
	points[:train_test_cutoff], points[:train_test_cutoff],
	epochs=50,
	batch_size=256,
	shuffle=True,
	validation_data=(points[train_test_cutoff:], points[train_test_cutoff:])
)

embedded_points = encoder.predict(points)

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

input_layer = Input(shape=(source_dim,))
encoder_layer_1 = Dense(64, activation="tanh")(input_layer)
encoder_layer_2 = Dense(32, activation="tanh")(encoder_layer_1)
encoder_layer_3 = Dense(32, activation="tanh")(encoder_layer_2)
choke_layer = Dense(target_dim, activation="tanh")(encoder_layer_3)
decoder_layer_3 = Dense(32, activation="tanh")(choke_layer)
decoder_layer_2 = Dense(32, activation="tanh")(decoder_layer_3)
decoder_layer_1 = Dense(64, activation="tanh")(decoder_layer_2)
output_layer = Dense(source_dim, activation="linear")(decoder_layer_1)

encoder = Model(input_layer, choke_layer)
autoencoder = Model(input_layer, output_layer)

train_test_cutoff = int(0.8 * num_points)

autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
autoencoder.fit(
	points[:train_test_cutoff], points[:train_test_cutoff],
	epochs=50,
	batch_size=256,
	shuffle=True,
	validation_data=(points[train_test_cutoff:], points[train_test_cutoff:])
)

embedded_points = encoder.predict(points)

############################

fig, ax = plt.subplots(figsize=(14.4, 10.8), dpi=100)
ax.scatter(embedded_points[:,0], embedded_points[:,1], c=color, cmap=plt.cm.Spectral, s=embedding_sp_rad**2, linewidths=embedding_sp_lw)
ax.set_title("\n".join(wrap("2D Embedding from Autoencoder", 60)))
plt.show()