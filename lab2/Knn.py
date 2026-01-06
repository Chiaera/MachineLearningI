import numpy as np

class Knn:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.x_train: np.ndarray = np.array([])
        self.y_train: np.ndarray = np.array([])
        self.k: int = n_neighbors
        self.x_test: np.ndarray = np.array([])

    def fit(self, x: np.ndarray, y: np.ndarray):
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y have different number of samples")
        
        self.x_train = x
        self.y_train = y


    def predict(self, x: np.ndarray):
        if self.x_train.size == 0 or self.y_train.size == 0:
            raise RuntimeError("Model is not fit")
        
        predictions = []
        for i in range(x.shape[0]):
            distances = np.linalg.norm(self.x_train - x[i], axis=1) # same as the Euclidean distance
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            prediction = np.bincount(k_labels).argmax()
            predictions.append(prediction)
            
        return np.array(predictions)

    def test(self, y_test, y_pred):
        return np.mean(y_test == y_pred)
