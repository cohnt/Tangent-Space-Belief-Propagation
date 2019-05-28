import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_message_1d(messages, t, s, ax, colormap=plt.cm.coolwarm):
	message = messages[t][s]
	num_points = len(message.pos)
	rgb = colormap(message.weights / max(message.weights))
	line_bottom = 0
	line_top = 1
	ax.vlines(message.pos.flatten(), line_bottom, line_top)
	ax.set_title("Message %d -> %d" % (t, s))