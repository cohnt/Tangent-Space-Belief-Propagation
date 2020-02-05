import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense
from keras.models import Model

class Autoencoder():
	def __init__(self, source_dim, target_dim, layer_sizes, layer_activations, choke_activation="tanh", output_activation="linear", optimizer="adadelta", loss="mean_squared_error"):
		self.source_dim = source_dim
		self.target_dim = target_dim
		self.layer_sizes = layer_sizes
		self.layer_activations = layer_activations
		self.choke_activation = choke_activation
		self.output_activation = output_activation

		self.num_encoder_layers = len(layer_sizes)

		self.encoder_layers = []
		self.encoder_layers.append(
			Input(shape=(self.source_dim,))
		)
		for i in range(0, self.num_encoder_layers):
			layer_size = self.layer_sizes[i]
			layer_activation = self.layer_activations[i]
			self.encoder_layers.append(
				Dense(layer_size, activation=layer_activation)(self.encoder_layers[i])
			)

		self.decoder_layers = []
		self.decoder_layers.append(
			Dense(self.target_dim, activation=self.choke_activation)(self.encoder_layers[-1])
		)
		for i in range(0, self.num_encoder_layers):
			layer_size = self.layer_sizes[-1-i]
			layer_activation = self.layer_activations[-1-i]
			self.decoder_layers.append(
				Dense(layer_size, activation=layer_activation)(self.decoder_layers[i])
			)
		self.output_layer = Dense(self.source_dim, activation=self.output_activation)(self.decoder_layers[-1])

		self.encoder = Model(self.encoder_layers[0], self.decoder_layers[0])
		self.autoencoder = Model(self.encoder_layers[0], self.output_layer)
		self.autoencoder.compile(optimizer=optimizer, loss=loss)

	def train(self, data, train_test_ratio=0.8, epochs=50, batch_size=256, shuffle=True):
		cutoff = int(len(data) * train_test_ratio)
		self.autoencoder.fit(
			data[:cutoff], data[:cutoff],
			epochs=epochs,
			batch_size=batch_size,
			shuffle=shuffle,
			validation_data=(data[cutoff:], data[cutoff:])
		)

	def embed(self, data):
		return self.encoder.predict(data)

	def reconstruct(self, data):
		return self.autoencoder.predict(data)