__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        if X.ndim > 2:
            self.X = np.reshape(X.astype(float), (X.shape[0] * X.shape[1], X.shape[2]))    
        else:
            self.X = X.astype(float)

        
    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.num_iter = 0
        self.old_centroids = np.zeros((self.K, self.X.shape[1]))
        self.centroids = np.zeros((self.K, self.X.shape[1]))
        if self.options['km_init'].lower() == 'first':
            # _, ix = np.unique(self.X, axis=0, return_index=True)
            # self.centroids = self.X[np.array(sorted(ix)[:self.K])]
            i = 0
            self.centroids[0] = self.X[0, :]
            for k in range(1, self.K):
                i += 1
                while (self.centroids[:k] == self.X[i, :]).all(1).any(0):
                    i += 1
                self.centroids[k] = self.X[i, :]
        elif self.options['km_init'].lower() == 'custom':
            for k in range(self.K):
                self.centroids[k, :] = k * 255 / (self.K - 1)
        elif self.options['km_init'].lower() == 'random':
            self.centroids[:] = self.X[np.random.randint(self.X.shape[0], size=self.K), :]


    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        #distance returns a PxK P are Px and K are the centroids and value is the distance
        dist = distance(self.X, self.centroids)
        self.labels = np.argmin(dist, 1)


    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        np.copyto(self.old_centroids, self.centroids)
        for i in range(self.K):
            x = self.X[self.labels == i, :]
            if x.size > 0:
                self.centroids[i, :] = np.mean(x, 0)
        
        # Without making use of for loops, seems to be slower
        # clustered_sums = np.zeros((self.K, self.X.shape[1]))
        # np.add.at(clustered_sums, self.labels, self.X)
        # clustered_counts = np.bincount(self.labels, minlength=self.K)
        # clustered_counts[clustered_counts == 0] = 1
        # self.centroids = clustered_sums / clustered_counts[:, np.newaxis]


    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        return np.allclose(self.centroids, self.old_centroids, atol=self.options['tolerance'], rtol=0.0)

        

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        self._init_centroids()
        self.get_labels()
        while not self.converges():
            self.num_iter += 1
            self.get_centroids()
            self.get_labels()
            # print(self.num_iter)

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
        # For each centroid, compute the
        """ dis_vals = np.zeros(self.K)
        for i in range(self.K):
            dis_vect = distance(self.X[self.labels == i], self.centroids[i][np.newaxis, :])
            dis_vals[i] = np.matmul(dis_vect.T, dis_vect)[0][0]
        
        return np.mean(dis_vals) """

        print((((self.X - self.centroids[self.labels, :]) ** 2).sum(axis=1)).mean())
        return (((self.X - self.centroids[self.labels, :]) ** 2).sum(axis=1)).mean()
        # return (((self.X - self.centroids[self.labels, :]) ** 2).sum(axis=1)**0.5).mean()
    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        #Steps:
        #1. Run the K-Means algorithm for each K from 2 to max_K on the data on which the KMeans object has been initialized
        #2. we will calculate the percentage of WCD decrement from the previous WCD for each K 
        #3. If 100− %DECk+1 is smaller than a 20% we take that k as the best K
        #4. If no K satisfies the condition, we take max_K as the best K
        # this function also trains the KMeans that means that it must update the parameters
        # we have to keep track of the parameters so whenever we find the k, we can return to the previous state
        #aquí també s'ha de fer el fit i actualitzar els parametres sempre utilitzant self
        
        SSE = []
        for k in range(2, max_K + 1):
            self.K = k
            self.fit()
            WCD = self.withinClassDistance()
            # print(WCD)
            SSE.append(WCD)

            if len(SSE) > 1 and SSE[-1] > SSE[-2] * 0.8:
                break

        self.K = self.K - 1
        self.fit()

        if self.options['verbose']:
            plt.subplot(133)
            plt.plot(SSE)
            plt.subplot(131)
            plt.imshow(self.X.reshape((80, 60, 3)) / 255)
            plt.axis('off')
            plt.subplot(132)
            plt.imshow((self.centroids[self.labels] / 255).reshape((80, 60, 3)))
            plt.axis('off')
            plt.show()

def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    #Idea for every centroid substract the X and then calculate the norm, in order to use matrix instead of for loops
    #do that for every centroid and then stack the results
    
    #X[:, np.newaxis]: (P, 3) to (P, 1, 3).
    #X[:, np.newaxis] - C: (K, 3) and X[:, np.newaxis] (P, 1, 3), NumPy applies broadcasting rules and subtracts
    # C from each 2D slice of X[:, np.newaxis] result:  (P, K, 3)
    return np.linalg.norm(X[:, np.newaxis] - C, axis=2) #axis=2 is the axis of the norm, the last one (depth)
    

def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """
    color_proba = utils.get_color_prob(centroids)
    ix = np.argmax(color_proba, axis=1)
    return list(utils.colors[ix])
