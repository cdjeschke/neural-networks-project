import copy
import random


class Exemplars(object):
    """
    Generates the exemplars for the numbers 0 through 8, returning them as either 10x10 matrices, or 
    vectors of length 100. 
    """

    def __init__(self):
        self.__zero = [
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
            [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
            [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
            [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
            [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
            [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
            [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
            [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        ]

        self.__one = [
            [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
        ]

        self.__two = [
            [-1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
            [-1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
            [-1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
            [-1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
            [-1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
            [-1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
        ]

        self.__three = [
            [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
            [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
            [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1],
            [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
            [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
            [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
        ]

        self.__four = [
            [-1, 1, 1, -1, -1, -1, -1, 1, 1, -1],
            [-1, 1, 1, -1, -1, -1, -1, 1, 1, -1],
            [-1, 1, 1, -1, -1, -1, -1, 1, 1, -1],
            [-1, 1, 1, -1, -1, -1, -1, 1, 1, -1],
            [-1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
            [-1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
        ]

        self.__five = [
            [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, 1, 1, -1, -1, -1, -1, -1],
            [-1, -1, -1, 1, 1, -1, -1, -1, -1, -1],
            [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
            [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1],
        ]

        self.__six = [
            [1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
            [1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
            [1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
            [1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1, -1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1, -1, -1],
            [1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
            [1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
        ]

        self.__seven = [
            [-1, -1, 1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, 1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
        ]

    @property
    def exemplars(self):
        return list([
            self.__zero,
            self.__one,
            self.__two,
            self.__three,
            self.__four,
            self.__five,
            self.__six,
            self.__seven
        ])

    @staticmethod
    def get_exemplars(as_matrices=False):
        x = Exemplars()

        if as_matrices:
            return x.exemplars
        else:
            exemplars = []
            for exemplar in x.exemplars:
                v_exemplar = [col for row in exemplar for col in row]
                exemplars.append(v_exemplar)

        return exemplars

    @staticmethod
    def to_matrix(v_exemplar):
        """
        Convert the vector representation of an exemplar to its 10x10 matrix form
        :param v_exemplar: list of 100 elements representing the exemplar
        :return: the exemplar in its 10x10 matrix form
        """
        assert(len(v_exemplar) == 100)
        return list([
            v_exemplar[0:10],
            v_exemplar[10:20],
            v_exemplar[20:30],
            v_exemplar[30:40],
            v_exemplar[40:50],
            v_exemplar[50:60],
            v_exemplar[60:70],
            v_exemplar[70:80],
            v_exemplar[80:90],
            v_exemplar[90:]
        ])

    @staticmethod
    def to_vector(exemplar):
        """
        Convert the matrix representation of the exemplar to its vector form (100 elements) 
        :param exemplar: 10x10 matrix form the exemplar
        :return: a 100 element list representing the vector for that exemplar
        """
        assert(len(exemplar) == 10)
        return list([col for row in exemplar for col in row])

    @staticmethod
    def add_noise(exemplar, p=.25):
        """
        Add random noise to an exemplar by flipping the value at each index (-1 => 1, 1 => -1) with probability p.
        :param exemplar: exemplar to add noise to
        :param p: probability with which to add noise
        :return: a copy of the exemplar with noise added
        """

        def flip_bit(x):
            i = random.uniform(0, 1)
            if i <= p:
                return x * -1
            else:
                return x

        noisy_exemplar = copy.deepcopy(exemplar)
        return [flip_bit(x) for x in noisy_exemplar]










