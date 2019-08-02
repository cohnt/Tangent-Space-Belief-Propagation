import numpy as np
import pandas as pd
import rospy
import matplotlib.pyplot as plt

from sensor_msgs.msg import LaserScan as LaserScanMsg
from time import sleep, time

data_file_name = "data.csv"
topic_name = "/base_scan"
node_name = "laser_scan_data_record_node"
rate = 0

try:
	open(data_file_name, "r")
	print "Error: %s already exists!" % data_file_name
	exit()
except IOError:
	pass

data_file = open(data_file_name, "w+")
t0 = time()

rospy.init_node(node_name)

iter_num = 1
while True:
	try:
		print "Recording data #%d" % iter_num
		data_message = rospy.wait_for_message(topic_name, LaserScanMsg)

		angles = np.linspace(data_message.angle_min, data_message.angle_max, len(data_message.ranges))
		ranges = np.array(data_message.ranges)

		# Replace infinity with range_max
		range_max = data_message.range_max
		ranges[ranges == np.inf] = range_max

		# Interpolate nan values in the middle
		ranges = pd.Series(ranges).interpolate().get_values()

		# Fix nan values on the end to the closest not-nan value
		ind = np.where(~np.isnan(ranges))[0]
		first, last = ind[0], ind[-1]
		ranges[:first] = ranges[first]
		ranges[last + 1:] = ranges[last]
		inter_ranges = ranges.copy()

	except KeyboardInterrupt:
		break

	######

	# orig_data = np.multiply([np.cos(angles), -np.sin(angles)], data_message.ranges).transpose()
	# inter_data = np.multiply([np.cos(angles), -np.sin(angles)], inter_ranges).transpose()

	# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
	# ax1.scatter(orig_data[:,0], orig_data[:,1])
	# ax2.scatter(inter_data[:,0], inter_data[:,1])
	# plt.show()
	# exit()

	######

	str = ','.join([`num` for num in ranges])
	data_file.write("%s\n" % str)

	sleep(rate)
	iter_num = iter_num + 1

data_file.close()