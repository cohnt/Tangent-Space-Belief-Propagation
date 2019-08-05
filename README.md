Run all scripts from the root of the repository as a module, i.e., to run ```pca_demo.py``` use the command ```python2 -m scripts/pca_demo```.

### TODO

* Implement least squares regression of best fit line on the embedded parameter vs true parameter graphs to quantitatively measure how good an embedding is.
* Make a diagram of a neighborhood within a graph (highlight or otherwise emphasize), and draw out the corresponding MRF network
* Method to draw the underlying manifold
* Find a recent publication relating to manifold learning
* Remove all mentions of "unfortunately" in the paper
* Write an abstract
* Come up with figures or other papers I can cite which demonstrate how existing algorithms don't work if there are short-circuit edges...
  * ISOMAP
  * LLE
  * LTSA
  * Manifold Charting?
  * Laplacian Eigenmaps?
  * Local Smoothing
  * Proximity Graphs
  * EIV Smoothing
  * BP-ISOMAP