import numpy as np
import matplotlib.pyplot as plt

plt.ion()

idx = 10

data_file_name = "data.csv"

data = np.genfromtxt(data_file_name, delimiter=",")

ranges = data[idx]

angle_min = np.pi * 220.0 / 180.0
angle_max = 0.0
angles = np.linspace(angle_min, angle_max, len(ranges))

# ##########
# # Ranges #
# ##########

# fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)
# ax.scatter(np.arange(len(ranges)), ranges)
# plt.show()

##########
# Points #
##########

# points = np.multiply([np.cos(angles), np.sin(angles)], ranges).transpose()
# fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)
# ax.scatter(points[:,0], points[:,1])
# plt.show()

fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)
paths = None
for i in range(len(data)):
	ranges = data[i]
	points = np.multiply([np.cos(angles), np.sin(angles)], ranges).transpose()
	try:
		paths.remove()
	except:
		pass
	paths = ax.scatter(points[:,0], points[:,1])

	fig.canvas.draw_idle()
	plt.pause(0.1)

plt.waitforbuttonpress()