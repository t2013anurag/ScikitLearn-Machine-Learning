from sklearn.neighbors import NearestNeighbors
import numpy as np
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]) #gives the euclidean distance
nbrs = NearestNeighbors(n_neighbors = 3, algorithm = 'ball_tree').fit(X)
#nbrs = NearestNeighbors(n_neighbors = 3, algorithm = 'kd_tree').fit(X)
# n_neighbors  = 3 will give 3 nearest neighbors.
distances, indices = nbrs.kneighbors(X)
print indices
print distances