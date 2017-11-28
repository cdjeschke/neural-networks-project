import numpy as np


class HopfieldNetwork(object):
    """
    Implements a Hopfield network for exemplar storage and retrieval.  The dimensions of the weight matrix
    for the network are infered from the exemplar vectors supplied during the initialization.
    """

    def __init__(self, v_exemplars, hebbian_test=False, debug=False):
        """
        Initialize a Hopefield network given a list of exemplars
        :param v_exemplars: 
        """
        # Initialize a logger for debug purposes
        def logger(msg):
            if debug:
                print msg

        self.__logger = logger

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

        if hebbian_test:
            print len(v_exemplars)
            self.__weight_matrix = np.true_divide(self.__weight_matrix, len(v_exemplars))


    @property
    def weight_matrix(self):
        return self.__weight_matrix

    def recall(self, v_p, asynchronous=False):
        """
        Recall an exemplar from the Hopfield network via F(Wv_p) where F is the hard limiting function and W is the 
        weight matrix of the network
        :param v_p: a noisy representation of the exemplar we want to recover
        :param asynchronous: method of recall
        :return: the retrieved exemplar
        """

        def hard_limiter(x):
            """
            Apply the hard limiting function. F(x) = 1 if x>=0 else, -1
            :param x: x
            :return: 1 or -1
            """
            return 1 if x >= 0 else -1

        self.__logger("Input vector is: {0}".format(v_p))
        if not asynchronous:
            self.__logger("Using synchronous recall.")

            # multiply this network's weight matrix by the transpose of the noisy exemplar
            v_pt = np.array([v_p]).transpose()
            results = np.dot(self.__weight_matrix, v_pt)
            p = results.transpose().flatten().tolist()
            p = map(hard_limiter, p)
            return p

        # Asynchronous execution model
        else:
            self.__logger("Using asynchronous recall.")
            x_s = v_p
            for i, row in enumerate(self.__weight_matrix):
                x_s_t = np.array([x_s]).transpose()
                self.__logger("Iteration {0}: {1} * {2}".format(i, row, x_s_t))
                results = np.dot(row, x_s_t)
                results = hard_limiter(results)
                self.__logger("x_s[{0}] is {1}".format(i, results))
                x_s[i] = results
            return x_s








