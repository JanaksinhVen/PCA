import numpy as np
import pandas as pd

#PCA model
class MY_PCA:
    def __init__(self, n_components):
        self.n_com = n_components
        self.X_std = []
        self.X_mean = []
        self.top_n_evector = []

    def fit(self, X_data):
        pass

    def transform(self, X_data):
        self.X_mean = np.mean(X_data,axis=0)
        self.X_std = np.std(X_data)
        std_data = (X_data-self.X_mean)/self.X_std
        X_cov = np.cov(std_data,rowvar=False)
        X_e_value, X_e_vector = np.linalg.eig(X_cov)
        X_evalue_evector = [(X_e_value[i], X_e_vector[:,i]) for i in range(len(X_e_value))]
        X_evalue_evector.sort(reverse=True, key=lambda eigen:eigen[0])
        
        self.top_n_evector = np.array([pair[1] for pair in X_evalue_evector[:self.n_com]]).T
        X_reduced_dim = np.dot(std_data, self.top_n_evector)
        P_col = [f"PC{x}" for x in range(self.n_com)]

        X_trans = pd.DataFrame(X_reduced_dim, columns=P_col)
        return X_trans, X_e_value

    def inv_transform(self,X_trans_data):
   
        X_reconstructed = np.dot(X_trans_data, self.top_n_evector.T)
        X_reconstructed = (X_reconstructed * self.X_std ) + self.X_mean
        
        return X_reconstructed


#KNN model as classifier
class KNN_Opt:
    def __init__(self, hy_param):
        self.k = hy_param[0]
       # self.encoder = hy_param[1]
        self.distance_matrix = hy_param[2]
        

        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        if self.distance_matrix == 'Euclidean':
            squared_diffs = (self.X_train - x)**2

            sum_squared_diffs = np.sum(squared_diffs, axis=1)

            distances = np.sqrt(sum_squared_diffs)

        elif self.distance_matrix == 'Manhattan':
            distances = [np.sum(np.abs(x - x_train)) for x_train in self.X_train]

            squared_diffs = np.abs(self.X_train - x)

            sum_squared_diffs = np.sum(squared_diffs, axis=1)

            distances = np.sqrt(sum_squared_diffs)

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        label_counts = {}
        for label in k_nearest_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        most_common_label = max(label_counts, key=label_counts.get)

        return most_common_label