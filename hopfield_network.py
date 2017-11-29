import math
import numpy as np


class HopfieldNetwork(object):
    """
    Implements a Hopfield network for exemplar storage and retrieval.  The dimensions of the weight matrix
    for the network are infered from the exemplar vectors supplied during the initialization.
    """

    def __init__(self, v_exemplars, learning_rule='Hebbian', debug=False):
        """
        Initialize a Hopfield Network given a list of exemplars and a specified learning rule
        :param v_exemplars: list of exemplars (list of vectors)
        :param learning_rule: "Hebbian" = Hebbian Learning Rule,  'Storkey' = Storkey Learning Rule
        :param debug:  Should debug output be printed?  Default is False.  
        """
        # Initialize a logger for debug purposes
        def logger(msg):
            if debug:
                print msg

        self.__logger = logger

        # Need at least 1 exemplar
        assert(len(v_exemplars[0]) >= 1)

        # Hebbian learning
        if learning_rule is "Hebbian":
            self.__hebbian_learning_rule(v_exemplars, scaled=False)
        elif learning_rule == "Storkey":
            self.__storkey_learning(v_exemplars)
        else:
            print "Unrecognized rule"
            # throw exception...

    def __hebbian_learning_rule(self, v_exemplars, scaled=False):
        """
        Implement the Hebb rule for learning the Hopfield network
        :param v_exemplars: exemplars
        :param scaled: should the weights be scaled by n=number of exemplars
        :return: initialized weight matrix
        """

        # Initialize the weight matrix
        self.__weight_matrix = np.outer(v_exemplars[0], v_exemplars[0])
        np.fill_diagonal(self.__weight_matrix, 0)

        # If more than 1 exemplar, update the weight matrix
        for m in range(1, len(v_exemplars)):
            weights_delta = np.outer(v_exemplars[m], v_exemplars[m])
            np.fill_diagonal(weights_delta, 0)
            self.__weight_matrix = np.add(self.__weight_matrix, weights_delta)

        if scaled:
            # Scale the weights
            self.__weight_matrix = np.true_divide(self.__weight_matrix, len(v_exemplars))

        # Estimate of capacity for a Hopfield Network trained via Hebbian learning
        self.__num_exemplars = len(v_exemplars)
        if self.__num_exemplars > 1:
            self.__capacity = (1.0 * self.__num_exemplars) / (2 * math.log(self.__num_exemplars))
        else:
            self.__capacity = 1

    def __storkey_learning(self, v_exemplars):
        """
        Implement storkey learning rule
        :param v_exemplars: 
        """

        # Start with empty matrix  (w_ij^0)
        self.__weight_matrix = np.zeros(shape=(len(v_exemplars[0]), len(v_exemplars[0])))

        # Incrementally include each exemplar
        for exemplar in v_exemplars:

            self.__logger("Adding Exemplar {0}".format(exemplar))
            weight_matrix_v = np.zeros(shape=(len(exemplar), len(exemplar)))

            def h(i, j, w, e, n):
                h_ij = sum([w[i][k] * e[k] for k in range(1, n) if k != i and k != j])
                return h_ij

            n = len(exemplar)
            for i in range(len(exemplar)):
                for j in range(len(exemplar)):

                    # skip i == j
                    if i == j:
                        continue

                    # 1st term = 1/n EV_i EV_j
                    t1 = (1.0 / n) * exemplar[i] * exemplar[j]

                    # 2nd term  = 1/n E^v_{i}H^v_{j,i}
                    t2 = (1.0 / n) * exemplar[i] * h(j, i, self.__weight_matrix, exemplar, n)

                    # 3rd term = 1/n h^v_{i,j} E^v_{j}
                    t3 = (1.0 / n) * h(i, j, self.__weight_matrix, exemplar, n) * exemplar[j]

                    weight_matrix_v[i][j] = self.__weight_matrix[i][j] + t1 - t2 - t3
                    self.__logger("w_{0},{1} = {2} + {3} - {4} - {5}".format(i, j, self.__weight_matrix[i][j], t1, t2, t3))

            # Update the weight matrix
            self.__weight_matrix = weight_matrix_v


    @property
    def num_exemplars(self):
        return self.__num_exemplars

    @property
    def capacity(self):
        return self.__capacity

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








