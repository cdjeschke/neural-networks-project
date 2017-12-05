import unittest

from lippmann_exemplars import LippmanExemplars


class TestExemplars(unittest.TestCase):
    """
    Unit test for retrieving the exemplars to use in assignment 2
    """

    def test_to_vector(self):
        """
        Confirm conversion from matrix to vector form works
        :return: 
        """
        zero = LippmanExemplars.get_exemplars(as_matrices=True)[0]
        v_zero = LippmanExemplars.to_vector(zero)
        assert(v_zero == [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1,
            -1, -1, 1, 1, -1, -1, 1, 1, -1, -1,
            -1, 1, 1, -1, -1, -1, -1, 1, 1, -1,
            -1, 1, 1, -1, -1, -1, -1, 1, 1, -1,
            -1, 1, 1, -1, -1, -1, -1, 1, 1, -1,
            -1, 1, 1, -1, -1, -1, -1, 1, 1, -1,
            -1, 1, 1, -1, -1, -1, -1, 1, 1, -1,
            -1, 1, 1, -1, -1, -1, -1, 1, 1, -1,
            -1, -1, 1, 1, -1, -1, 1, 1, -1, -1,
            -1, -1, -1, 1, 1, 1, 1, -1, -1, -1])

    def test_to_matrix(self):
        """
        Confirm conversion from vector to matrix form works
        :return: 
        """
        v_four = LippmanExemplars.get_exemplars()[4]
        four = LippmanExemplars.to_matrix(v_four)
        assert(four == [
            [-1, -1, -1, 1, -1, -1, -1, -1, -1, 1],
            [-1, -1, -1, 1, -1, -1, -1, -1, -1, 1],
            [-1, -1, -1, 1, -1, -1, -1, -1, -1, 1],
            [-1, -1, -1, 1, -1, -1, -1, -1, -1, 1],
            [-1, -1, -1, 1, -1, -1, -1, -1, -1, 1],
            [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
        ])


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestExemplars)
    unittest.TextTestRunner(verbosity=2).run(suite)