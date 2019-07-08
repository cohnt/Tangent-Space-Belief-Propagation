import numpy as np
import matplotlib.pyplot as plt
from plot_neighbors import plot_neighbors_2d, plot_neighbors_3d
from sklearn.neighbors import kneighbors_graph

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from datasets.dim_2.arc_curve import make_arc_curve
from datasets.dim_2.o_curve import make_o_curve
from datasets.dim_2.s_curve import make_s_curve
from datasets.dim_3.helix_curve import make_helix_curve
from datasets.dim_3.s_sheet import make_s_sheet

#############
# arc_curve #
#############
fig, ax = plt.subplots()
data, color = make_arc_curve(500, 0.001)
graph = kneighbors_graph(data, 12, mode="distance", n_jobs=-1)
plot_neighbors_2d(data, color, graph, ax)

###########
# o_curve #
###########
fig, ax = plt.subplots()
data, color = make_o_curve(500, 0.0005)
graph = kneighbors_graph(data, 12, mode="distance", n_jobs=-1)
plot_neighbors_2d(data, color, graph, ax)

###########
# s_curve #
###########
fig, ax = plt.subplots()
data, color = make_s_curve(500, 0.001)
graph = kneighbors_graph(data, 12, mode="distance", n_jobs=-1)
plot_neighbors_2d(data, color, graph, ax)

###############
# helix_curve #
###############
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
data, color = make_helix_curve(500, 0.001)
graph = kneighbors_graph(data, 12, mode="distance", n_jobs=-1)
plot_neighbors_3d(data, color, graph, ax)

###########
# s_sheet #
###########
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
data, color = make_s_sheet(500, 0.001)
graph = kneighbors_graph(data, 12, mode="distance", n_jobs=-1)
plot_neighbors_3d(data, color, graph, ax)

plt.show()