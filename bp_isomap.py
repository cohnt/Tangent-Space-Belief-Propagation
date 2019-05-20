import numpy as np
import matplotlib.pyplot as plt

################
# Load Dataset #
################
from datasets.dim_2.s_curve import make_s_curve

points, color = make_s_curve(500, 0.001)

#######################
# k-Nearest-Neighbors #
#######################
from sklearn.neighbors import kneighbors_graph

neighbor_graph = kneighbors_graph(points, 12, mode="distance", n_jobs=-1)

####################
# Initialize Graph #
####################
from utils import sparseMatrixToDict

neighbor_dict = sparseMatrixToDict(neighbor_graph)

#######################
# Initialize Messages #
#######################

# TODO

###################
# Message Passing #
###################

# TODO