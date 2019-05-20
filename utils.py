import numpy as np

def sparseMatrixToDict(mat):
	return {i: [j for j, adjacent in enumerate(row) if adjacent] for i, row in enumerate(mat.toarray())}