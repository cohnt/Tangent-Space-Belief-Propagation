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
	ax.vlines(belief.pos.flatten(), line_bottom, line_top, colors=rgb, linewidth=line_width, linestyles='solid')

	if show_mle:
		ind = np.argmax(belief.weights)
		pos = belief.pos.flatten()[ind]
		label_text = "MLE=%f" % pos
		ax.axvline(pos, line_bottom, line_top, color="green", linewidth=mle_line_width, linestyle="--", dash_capstyle="round")
		ax.text(pos, 0.1, label_text)

	if show_mean:
		pos = np.average(belief.pos.flatten(), weights=belief.weights)
		label_text = "Mean=%f" % pos
		ax.axvline(pos, line_bottom, line_top, color="black", linewidth=mean_line_width, linestyle="--", dash_capstyle="round")
		ax.text(pos, 0.05, label_text)

def plot_mle_1d(beliefs, colors, ax, colormap=plt.cm.spectral, line_width=1):
	num_points = len(beliefs)
	points = np.zeros(num_points)
	for i in range(num_points):
		ind = np.argmax(beliefs[i].weights)
		points[i] = beliefs[i].pos.flatten()[ind]
	line_bottom = 0
	line_top = 1
	rgb = colormap(colors)
	ax.vlines(points, line_bottom, line_top, colors=rgb, linewidth=line_width, linestyles='solid')

def plot_mean_1d(beliefs, colors, ax, colormap=plt.cm.spectral, line_width=1):
	num_points = len(beliefs)
	points = np.zeros(num_points)
	for i in range(num_points):
		points[i] = np.average(beliefs[i].pos.flatten(), weights=beliefs[i].weights)
	line_bottom = 0
	line_top = 1
	rgb = colormap(colors)
	ax.vlines(points, line_bottom, line_top, colors=rgb, linewidth=line_width, linestyles='solid')