import copy
import random


class LippmanExemplars(object):
    """
    Generates the eight exemplar patterns used in Lippmann's paper. 
    Each exemplar is a 12 x 10 matrix with:
        1 = a black pixel
        -1 = a white pixel
    """
    def __init__(self):
        # 12 x 10
        self.__exemplar_1 = [
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
            [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
            [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
            [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
            [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
            [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
            [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
            [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
            [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
            [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        ]

        # 12 x 10
        self.__exemplar_2 = [
            [-1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
            [-1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
            [-1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
            [-1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
            [-1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
            [-1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
            [-1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
            [-1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
            [-1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
            [-1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
            [-1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
            [-1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
        ]

        # 12 x 10
        self.__exemplar_3 = [
            [-1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
            [-1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
            [-1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
            [-1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
            [-1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
            [-1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
        ]

        # 12 x 10
        self.__exemplar_4 = [
            [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
            [-1, -1, 1, 1, 1, 1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
            [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1],
            [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
            [-1, -1, 1, 1, 1, 1, 1, 1, 1, -1],
            [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
        ]

        # 12 x 10
        self.__exemplar_5 = [
            [-1, 1, 1, -1, -1, -1, -1, 1, 1, -1],
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
            [-1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
        ]

        # 12 x 10
        self.__exemplar_6 = [
            [-1, 1, 1, 1, 1, 1, 1, -1, -1, -1],
            [-1, 1, 1, 1, 1, 1, 1, -1, -1, -1],
            [-1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 1, 1, 1, 1, 1, 1, -1, -1, -1],
            [-1, 1, 1, 1, 1, 1, 1, -1, -1, -1],
            [-1, 1, 1, -1, -1, 1, 1, -1, -1, -1],
            [-1, 1, 1, -1, -1, 1, 1, -1, -1, -1],
            [-1, 1, 1, -1, -1, 1, 1, -1, -1, -1],
            [-1, 1, 1, 1, 1, 1, 1, -1, -1, -1],
            [-1, 1, 1, 1, 1, 1, 1, -1, -1, -1],
        ]

        # 12 x 10
        self.__exemplar_7 = [
            [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
            [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
            [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
            [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
            [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
            [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        ]

        # 12 x 10
        self.__exemplar_8 = [
            [-1, -1, -1, -1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
            [-1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
            [-1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
            [-1, -1, -1, -1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
            [-1, -1, -1, -1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, 1, 1, 1, 1, 1, 1],
        ]

    @property
    def exemplars(self):
        return list([
            self.__exemplar_1,
            self.__exemplar_2,
            self.__exemplar_3,
            self.__exemplar_4,
            self.__exemplar_5,
            self.__exemplar_6,
            self.__exemplar_7,
            self.__exemplar_8
        ])

    @staticmethod
    def get_exemplars(as_matrices=False):
        x = LippmanExemplars()

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
        Convert the vector representation of an exemplar to its 12 x 10 matrix form
        :param v_exemplar: the vector representation of the exemplar
        :return: the exemplar as a 12 x 10 matrix
        """
        assert(len(v_exemplar) == 120)
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
            v_exemplar[90:100],
            v_exemplar[100:110],
            v_exemplar[110:120],
        ])

    @staticmethod
    def to_vector(exemplar):
        """
        Convert the matrix representation of the exemplar to its vector form (100 elements) 
        :param exemplar: 12x10 matrix form the exemplar
        :return: a 120 element list representing the vector for that exemplar
        """
        assert(len(exemplar) == 12)
        return list([col for row in exemplar for col in row])

    @staticmethod
    def add_noise(exemplar, p=.25):
        """
        Add random noise to an exemplar by flipping the value at each index (-1 => 1, 1 => -1) with probability p.
        :param exemplar: exemplar to add noise to
        :param p: probability with which to add noise
        :return: a copy of the exemplar with noise added
        """
        assert(len(exemplar) == 120)

        def flip_bit(x):
            i = random.uniform(0, 1)
            if i <= p:
                return x * -1
            else:
                return x

        noisy_exemplar = copy.deepcopy(exemplar)
        return [flip_bit(x) for x in noisy_exemplar]










