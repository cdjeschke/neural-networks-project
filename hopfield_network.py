import copy
import math
import numpy as np


class HopfieldNetwork(object):
    """
    Implements a Hopfield network for exemplar storage and retrieval.  The dimensions of the weight matrix
    for the network are infered from the exemplar vectors supplied during the initialization.
    """

    def __init__(self, v_exemplars, learning_rule='Hebb', debug=False):
        """
        Initialize a Hopfield Network given a list of exemplars and a specified learning rule
        :param v_exemplars: list of exemplars (list of vectors)
        :param learning_rule: "Hebb" = Hebbian Learning Rule,  'Storkey' = Storkey Learning Rule
        :param debug:  Should debug output be printed?  Default is False.  
        """
        # Initialize a logger for debug purposes
        def logger(msg):
            if debug:
                print msg

        self.__logger = logger

        # Need at least 1 exemplar
        assert(len(v_exemplars[0]) >= 1)

        # All exemplars should be same length
        n = len(v_exemplars[0])
        for exemplar in v_exemplars:
            assert(len(exemplar) == n)
        self.__exemplars = v_exemplars

        # The number of neurons is the number of elements in each exemplar
        self.__num_neurons = len(self.__exemplars[0])

        # Hebbian learning
        if learning_rule is "Hebb":
            self.__hebbian_learning_rule(v_exemplars)
        elif learning_rule == "Storkey":
            self.__storkey_learning(v_exemplars)
        else:
            print "Unrecognized rule"
            # throw exception...

    def __hebbian_learning_rule(self, v_exemplars):
        """
        Implement the Hebb rule for learning the Hopfield network
        :param v_exemplars: exemplars
        :return: initialized weight matrix
        """
        # Start with an empty weights matrix
        self.__weight_matrix = np.zeros(shape=(self.__num_neurons, self.__num_neurons))

        # Initialize the weight matrix
        for exemplar in v_exemplars:
            n = len(exemplar)
            # Hebbian Learning Rule:  w_ij_new = w_ij_current + (1/num neurons) * e_i * e_j
            # where i & j are neuron indices & indices into the exemplar
            weights_delta = np.outer(exemplar, exemplar)
            weights_delta = (1.0 /n) * weights_delta

            np.fill_diagonal(weights_delta, 0)
            self.__weight_matrix = np.add(self.__weight_matrix, weights_delta)

        # Capacity for a Hopfield Network trained via Hebbian learning
        if self.__num_neurons > 1:
            self.__capacity = (1.0 * self.__num_neurons) / (2 * math.log(self.__num_neurons))
        else:
            self.__capacity = 1

    def __storkey_learning(self, v_exemplars):
        """
        Implement the Storkey Learning Rule
        :param v_exemplars: 
        """

        # Start with empty matrix  (w_ij^0)
        self.__weight_matrix = np.zeros(shape=(self.__num_neurons, self.__num_neurons))

        # Incrementally include each exemplar
        for exemplar in v_exemplars:

            self.__logger("Adding Exemplar {0}".format(exemplar))
            weight_matrix_v = np.zeros(shape=(self.__num_neurons, self.__num_neurons))

            def h(i, j, w, e, n):
                h_ij = sum([w[i][k] * e[k] for k in range(1, n) if k != i and k != j])
                return h_ij

            for i in range(len(exemplar)):
                for j in range(len(exemplar)):

                    # skip i == j
                    if i == j:
                        continue

                    # 1st term = 1/n EV_i EV_j
                    t1 = (1.0 / self.__num_neurons) * exemplar[i] * exemplar[j]

                    # 2nd term  = 1/n E^v_{i}H^v_{j,i}
                    t2 = (1.0 / self.__num_neurons) * exemplar[i] * \
                         h(j, i, self.__weight_matrix, exemplar, self.__num_neurons)

                    # 3rd term = 1/n h^v_{i,j} E^v_{j}
                    t3 = (1.0 / self.__num_neurons) * h(i, j, self.__weight_matrix, exemplar, self.__num_neurons) * \
                         exemplar[j]

                    weight_matrix_v[i][j] = self.__weight_matrix[i][j] + t1 - t2 - t3
                    self.__logger("w_{0},{1} = {2} + {3} - {4} - {5}".format(i, j, self.__weight_matrix[i][j],
                                                                             t1, t2, t3))

            # Update the weight matrix
            self.__weight_matrix = weight_matrix_v

            # Capacity for a Hopfield Network trained using Storkey
            if self.__num_neurons > 1:
                self.__capacity = (1.0 * self.__num_neurons) / math.sqrt(2 * math.log(self.__num_neurons))
            else:
                self.__capacity = 1

    @property
    def num_neurons(self):
        return self.__num_neurons

    @property
    def num_exemplars(self):
        return len(self.__v_exemplars)

    @property
    def capacity(self):
        return self.__capacity

    @property
    def weight_matrix(self):
        return self.__weight_matrix

    def synchronous_recall(self, v_p, max_iterations=10):
        """
        Recall an exemplar from the Hopfield network via F(Wv_p) where F is the hard limiting function and W is the 
        weight matrix of the network
        :param v_p: a noisy representation of the exemplar we want to recover
        :return: the retrieved exemplar
        """

        def hard_limiter(x):
            """
            Apply the hard limiting function. F(x) = 1 if x>=0 else, -1
            :param x: x
            :return: 1 or -1
            """
            return 1 if x >= 0 else -1

        self.__logger("Using synchronous recall.")
        self.__logger("Input vector is: {0}".format(v_p))

        # results to return
        results = list()

        x_s = v_p
        converged = False
        i = 0
        while not converged and i < max_iterations:
            x_s_prev = copy.deepcopy(x_s)

            x_s = np.dot(self.__weight_matrix, np.array([x_s]).transpose())
            x_s = map(hard_limiter, x_s)

            results.append(x_s)

            self.__logger("x_s: {0}, x_s_prev: {1}".format(x_s, x_s_prev))
            # Convergence when the state of the neurons (x_s) is unchanged
            if x_s == x_s_prev:
                converged = True

        return results

    def asynchronous_recall(self, v_p):
        """
        Recall an exemplar using asynchronous updating.
        :param v_p: noisy vector we want to recall from 
        :return: [(i, result)]
        """

        def hard_limiter(x):
            """
            Apply the hard limiting function. F(x) = 1 if x>=0 else, -1
            :param x: x
            :return: 1 or -1
            """
            return 1 if x >= 0 else -1

        self.__logger("Using asynchronous recall.")
        self.__logger("Input vector is: {0}".format(v_p))

        # results to return
        results = list()

        x_s_prev = v_p
        converged = False
        while not converged:
            x_s = copy.deepcopy(x_s_prev)

            for i, row in enumerate(self.__weight_matrix):
                self.__logger("Evaluating at neuron [{0}]".format(i))
                x_i = np.dot(row, np.array([x_s]).transpose())
                x_i = hard_limiter(x_i)
                self.__logger("x_s[{0}] updated to {1}".format(i, x_i))
                x_s[i] = x_i
            results.append(x_s)

            self.__logger("x_s: {0}, x_s_prev: {1}".format(x_s, x_s_prev))
            # Convergence when the state of the neurons (x_s) is unchanged
            if x_s == x_s_prev:
                converged = True

            x_s_prev = x_s

        return results

