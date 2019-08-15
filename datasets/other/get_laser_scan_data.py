import numpy as np
import pandas as pd
import rospy
import matplotlib.pyplot as plt

from sensor_msgs.msg import LaserScan as LaserScanMsg
from sensor_msgs.msg import Image as ImageMsg
from time import sleep, time

from cv_bridge import CvBridge, CvBridgeError
import cv2

data_file_name = "data.csv"
topic_name = "/base_scan"
node_name = "laser_scan_data_record_node"
rate = 0.05
density = 2
image_rate = 10

bridge = CvBridge()
image_topic_name = "/head_camera/rgb/image_raw"
image_dir = "images/"

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

		if iter_num % image_rate == 0:
			image_message = rospy.wait_for_message(image_topic_name, ImageMsg)

			try:
				cv2_img = bridge.imgmsg_to_cv2(image_message, "bgr8")
			except CvBridgeError, e:
				print e
			else:
				cv2.imwrite(image_dir + ("iter%s.png" % str(iter_num).zfill(3)), cv2_img)

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

		# Downsample
		ranges = ranges[0::density]

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

	data = ','.join([`num` for num in ranges])
	data_file.write("%s\n" % data)

	sleep(rate)
	iter_num = iter_num + 1

data_file.close()
print "Data dimension: %d" % len(ranges)