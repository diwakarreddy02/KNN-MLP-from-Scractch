# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: Venkata Diwakar Reddy Kashireddy -- vkashir
#
# Based on skeleton code by CSCI-B 551 Fall 2022 Course Staff

import numpy as np
import math
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """
    min_max = [()]

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        
        self._X = X 
        self._y = y

        pass

        # raise NotImplementedError('This function must be implemented by the student.')

    def get_neighbours(self, n_neighbors, test_row):
        
        all_dist = []
        neighbors = []
        i = 0
        for row in self._X:
            dist = self._distance(row, test_row)
            row = np.append(row,self._y[i])
            i += 1
            all_dist.append((row, dist))

        all_dist.sort(key=lambda tup: tup[1])
        # print(all_dist)
        for k in range(n_neighbors):
            neighbors.append(all_dist[k][0])

        # print("\n",neighbors)
        return neighbors

    def get_predictValues(self, row):
        predicted_values = []
        neighbors = self.get_neighbours(self.n_neighbors, row)
        output_values = []
        for i in neighbors:
            output_values.append(i[-1])
        predicted_values = max(set(output_values), key=output_values.count)
        # print(predicted_values)
        return predicted_values

    def get_distance(self, train_row):
        out = []
        i = 0
        for row in self._X:
            dist = self._distance(train_row, row)
            row = np.append(row,self._y[i])
            i += 1
            if dist == 0:
                out.append((row, math.inf))
            else : #To give more preference to the nearer neighbors, this weighting method is used. 
                out.append((row, 1/dist))
        out.sort(key=lambda a: a[1], reverse = True)
        return out

    def get_max(self, test_row):

        neighbors = []
        output_values = []
        dist = self.get_distance(test_row)
        #print(dist)
        for k in range(self.n_neighbors):
            neighbors.append(dist[k][0])
        for i in neighbors:
            output_values.append(i[-1])
        
        predicted_values = max(set(output_values), key=output_values.count)
        # print(output_values,"pred: ",predicted_values)
        return predicted_values
            
    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        predicted_class = []
        if self.weights == 'uniform':
            class_values = list()
            for row in X:
                output = self.get_predictValues(row)
                class_values.append(output)

            result = np.array(class_values)
            return result

        elif self.weights == 'distance':
            class_values = []
            for test_row in X:
                output = self.get_max(test_row)
                class_values.append(output)
            result = np.array(class_values)
            return result  
