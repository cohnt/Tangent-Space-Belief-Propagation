Run all scripts from the root of the repository as a module, i.e., to run ```pca_demo.py``` use the command ```python2 -m scripts/pca_demo```.

### TODO

* Implement least squares regression of best fit line on the embedded parameter vs true parameter graphs to quantitatively measure how good an embedding is.
* Generalize resampling to work in any number of dimensions. Idea: make a rotation about the subspace about e_3, ..., e_n, but in a random basis. That is, take a uniform sample B from SO(n), use that to transform the basis, rotate with the specific rotation, and then transform the basis back using the inverse of B (i.e. the transpose because B is orthogonal).
* Make a diagram of a neighborhood within a graph (highlight or otherwise emphasize), and draw out the corresponding MRF network
* Method to draw the underlying manifold
* Find a recent publication relating to manifold learning
* Remove all mentions of "unfortunately" in the paper
* Write an abstract
* Come up with figures or other papers I can cite which demonstrate how existing algorithms don't work if there are short-circuit edges