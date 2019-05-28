import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from utils import Belief

def plot_belief_1d(belief, ax, line_width=1, colormap=plt.cm.coolwarm, show_mle=False, mle_line_width=4, show_mean=False, mean_line_width=4):
	num_points = len(belief.pos)
	rgb = colormap(belief.weights / max(belief.weights))
	line_bottom = 0
	line_top = 1
	ax.vlines(belief.pos.flatten(), line_bottom, line_top, colors=rgb, linestyles='solid')

	if show_mle:
		ind = np.argmax(belief.weights)
		pos = belief.pos.flatten()[ind]
		label_text = "MLE=%f" % pos
		ax.axvline(pos, line_bottom, line_top, color="green", linewidth=mle_line_width, linestyle="--", dash_capstyle="round")
		ax.text(pos, 0.1, label_text)

	if show_mean:
		pos = np.average(belief.pos.flatten(), weights=belief.weights)
		label_text = "Mean=%f" % pos
		ax.axvline(pos, line_bottom, line_top, color="black", line_width=mean_line_width, linestyle="--", dash_capstyle="round")
		ax.text(pos, 0.05, label_text)