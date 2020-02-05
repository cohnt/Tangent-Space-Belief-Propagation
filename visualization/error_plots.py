import matplotlib.pyplot as plt

def relativeErrorBarChart(ax, dict):
	rect = ax.bar(range(len(dict)), list(dict.values()), align='center')
	ax.set_title("Reconstruction Error by Manifold Learning Algorithm")
	ax.set_ylabel("Error")
	ax.set_xticks(range(len(dict)))
	ax.set_xticklabels(list(dict.keys()))
	ax.tick_params(axis='x', which='major', labelsize=10)

	autolabel(ax, rect)

def autolabel(ax, rects, xpos='center'):
	xpos = xpos.lower()
	ha = {'center': 'center', 'right': 'left', 'left': 'right'}
	offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

	for rect in rects:
		height = rect.get_height()
		ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
		        '{}'.format(height), ha=ha[xpos], va='bottom')
