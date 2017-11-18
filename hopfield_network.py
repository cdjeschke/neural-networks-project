import numpy as np


class HopfieldNetwork(object):
    """
    Implements a Hopfield network for exemplar storage and retrieval.  The dimensions of the weight matrix
    for the network are infered from the exemplar vectors supplied during the initialization.
    """

    def __init__(self, v_exemplars):
        """
        Initialize a Hopefield network given a list of exemplars
        :param v_exemplars: 
        """
        # Need at least 1 exemplar
        assert(len(v_exemplars[0]) >= 1)

        # Initialize the weight matrix
        self.__weight_matrix = np.outer(v_exemplars[0], v_exemplars[0])
        np.fill_diagonal(self.__weight_matrix, 0)

        # If more than 1 exemplar, update the weight matrix
        for m in range(1, len(v_exemplars)):
            weights_delta = np.outer(v_exemplars[m], v_exemplars[m])
            np.fill_diagonal(weights_delta, 0)
            self.__weight_matrix = np.add(self.__weight_matrix, weights_delta)

    @property
    def weight_matrix(self):
        return self.__weight_matrix

    def recall(self, v_p):
        """
        Recall an exemplar from the Hopfield network via F(Wv_p), where:
            - F is the hard limiting function
            - W is the weight matrix of the network
            - v_p is a "noisy" representation of the exemplar we're looking for
        :param v_p: noisy representation of the exemplar as a list (1-d array)
        :return: the found exemplar p
        """
        v_pt = np.array([v_p]).transpose()

        # multiply this network's weight matrix by the transpose of the noisy exemplar
        results = np.dot(self.__weight_matrix, v_pt)
        p = results.transpose().flatten().tolist()

        # Apply the hard limiting function.  F(x) = 1 if x >= 0, else -1
        def hard_limiter(x):
            return 1 if x >= 0 else -1

        p = map(hard_limiter, p)
        return p







