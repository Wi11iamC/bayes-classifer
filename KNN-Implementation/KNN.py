import numpy as np
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_prediction = [self._predict(x) for x in X]
        return np.array(y_prediction)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        dist = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(dist)[: self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # return the most common class label
        
        # count Neighbors
        N_counter = {}
        for word in k_neighbor_labels:
            if word in N_counter:
                N_counter[word] += 1
            else:
                N_counter[word] = 1
 
        popular_N = sorted(N_counter, key = N_counter.get, reverse = True)[0]
        return popular_N