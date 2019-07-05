import numpy as np
import matplotlib.pyplot as plt
import time

from datasets.dim_2.o_curve import make_o_curve
from sklearn.neighbors import kneighbors_graph
from visualization.plot_neighbors import plot_neighbors_2d
from sklearn.decomposition import PCA
from utils import sparseMaximum

neighbors_k = 12
num_points = 400

pca = PCA(n_components=1)

points, color = make_o_curve(num_points, 0.0001)
neighbor_graph = kneighbors_graph(points, neighbors_k, mode="distance", n_jobs=-1)
neighbor_graph = sparseMaximum(neighbor_graph, neighbor_graph.T)
