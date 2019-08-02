import numpy as np

data_file_name = "datasets/other/data.csv"

def make_laser_scan_curve():
	data = np.genfromtxt(data_file_name, delimiter=",")
	indices = np.arange(len(data))
	return data, indices