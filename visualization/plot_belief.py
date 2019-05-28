import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from utils import Belief

def plot_belief_1d(belief, ax, line_width=1, colormap=plt.cm.coolwarm):
	num_points = len(belief.pos)
	rgb = colormap(belief.weights / max(belief.weights))
	line_bottom = 0
	line_top = 1
	ax.vlines(belief.pos.flatten(), line_bottom, line_top, colors=rgb, linestyles='solid')