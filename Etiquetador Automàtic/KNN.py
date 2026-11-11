__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist
from collections import Counter



class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        if train_data.ndim > 2:
            self.train_data = np.reshape(train_data.astype(float), (train_data.shape[0], train_data.shape[1]*train_data.shape[2]))   
        else:
            self.train_data = train_data.astype(float)

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        if test_data.ndim > 2:
            test_data = np.reshape(test_data.astype(float), (test_data.shape[0], test_data.shape[1]*test_data.shape[2]))   
        else:
            test_data = test_data.astype(float)
        
        # A n_train*n_data distance matrix is returned 
        # dist(u=test[i], v=train[j]) is computed and stored in the ij-th entry.
        distances = cdist(test_data, self.train_data) # This computes the Euclidean distance between two points
       
        #Calculate class
        #get the k closest train samples to the test sample
        #for every test sample get the array of distances to that sample sort and get the best k ones
        neighbors_indices = np.argsort(distances, axis=1)[:, :k]
        self.neighbors = self.labels[neighbors_indices]


    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        classes = np.empty(self.neighbors.shape[0], dtype=object)

        for i, row in enumerate(self.neighbors):
            # Count the occurrences of each label in the row
            label_counts = Counter(row)
            most_frequent_label = max(label_counts, key=label_counts.get)
            classes[i] = most_frequent_label

        return classes
    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """
        self.get_k_neighbours(test_data, k)
        return self.get_class()
