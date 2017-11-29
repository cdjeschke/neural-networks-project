import numpy as np
import numpy.testing as npt
import unittest

from hopfield_network import HopfieldNetwork


class TestHopfieldNetwork(unittest.TestCase):
    """
    Unit test for retrieving the exemplars to use in assignment 2
    """

    def test_init_hebbian(self):
        v_one = [1, -1, -1, -1, 1, -1, -1, -1, 1]
        network = HopfieldNetwork([v_one])
        expected = np.array([
            [0, -1, -1, -1, 1, -1, -1, -1, 1],
            [-1, 0, 1, 1, -1, 1, 1, 1, -1],
            [-1, 1, 0, 1, -1, 1, 1, 1, -1],
            [-1, 1, 1, 0, -1, 1, 1, 1, -1],
            [1, -1, -1, -1, 0, -1, -1, -1, 1],
            [-1, 1, 1, 1, -1, 0, 1, 1, -1],
            [-1, 1, 1, 1, -1, 1, 0, 1, -1],
            [-1, 1, 1, 1, -1, 1, 1, 0, -1],
            [1, -1, -1, -1, 1, -1, -1, -1, 0]
        ], np.int64)
        npt.assert_equal(network.weight_matrix, expected)
        self.assertTrue(network.num_exemplars == 1)
        self.assertTrue(network.capacity == 1)

    def test_init_hebbian_2(self):
        """
        Test the initialization of a Hopfield network using 2 exemplars
        """
        v_one = [1, -1, -1, -1, 1, -1, -1, -1, 1]
        v_two = [-1, -1, -1, 1, 1, 1, -1, -1, -1]
        network = HopfieldNetwork([v_one, v_two])
        expected = np.array([
            [0, 0, 0, -2, 0, -2, 0, 0, 2],
            [0, 0, 2, 0, -2, 0, 2, 2, 0],
            [0, 2, 0, 0, -2, 0, 2, 2, 0],
            [-2, 0, 0, 0, 0, 2, 0, 0, -2],
            [0, -2, -2, 0, 0, 0, -2, -2, 0],
            [-2, 0, 0, 2, 0, 0, 0, 0, -2],
            [0, 2, 2, 0, -2, 0, 0, 2, 0],
            [0, 2, 2, 0, -2, 0, 2, 0, 0],
            [2, 0, 0, -2, 0, -2, 0, 0, 0]
        ], np.int64)
        npt.assert_equal(network.weight_matrix, expected)

    def test_init_hebbian_3(self):
        """
        Test the initialization of a Hopfield network using 2 exemplars
        """
        v_one = [1, -1, -1, -1, 1, -1, -1, -1, 1]
        v_two = [-1, -1, -1, 1, 1, 1, -1, -1, -1]
        v_three = [-1, -1, 1, -1, -1, 1, -1, -1, 1]
        network = HopfieldNetwork([v_one, v_two, v_three])
        expected = np.array([
            [0, 1, -1, -1, 1, -3, 1, 1, 1],
            [1, 0, 1, 1, -1, -1, 3, 3, -1],
            [-1, 1, 0, -1, -3, 1, 1, 1, 1],
            [-1, 1, -1, 0, 1, 1, 1, 1, -3],
            [1, -1, -3, 1, 0, -1, -1, -1, -1],
            [-3, -1, 1, 1, -1, 0, -1, -1, -1],
            [1, 3, 1, 1, -1, -1, 0, 3, -1],
            [1, 3, 1, 1, -1, -1, 3, 0, -1],
            [1, -1, 1, -3, -1, -1, -1, -1, 0]
        ])
        npt.assert_equal(network.weight_matrix, expected)

    def test_init_storkey_simple(self):
        """
        Test that the weights matrix for a Hopfield Network trained using the Storkey Learning Rule
        is calculated correctly.
        """
        expected_weights = np.array([
            [0, .3333333, .3333333],
            [.3333333, 0, .3333333],
            [.3333333, .3333333, 0]
        ])

        v_one = [1, 1, 1]
        network = HopfieldNetwork([v_one], learning_rule="Storkey")
        npt.assert_almost_equal(network.weight_matrix, expected_weights)

    def test_init_storkey_simple_2(self):
        """
        Test that the weights matrix for a Hopfield Network trained using the Storkey Learning Rule
        is calculated correctly with 2 exemplars.
        """
        expected_weights = np.array([
            [0, 0, .8888888],
            [0, 0, 0],
            [.8888888, 0, 0]
        ])

        v_one = [1, 1, 1]
        v_two = [1, -1, 1]
        network = HopfieldNetwork([v_one, v_two], learning_rule="Storkey")
        npt.assert_almost_equal(network.weight_matrix, expected_weights)

    def test_init_storkey(self):
        """
        Initialization using the Storkey Learning Rule
        :return: 
        """
        v_one = [1, -1, -1, -1, 1, -1, -1, -1, 1]

        network = HopfieldNetwork([v_one], learning_rule="Storkey")
        expected = np.array([
            [0, -1, -1, -1, 1, -1, -1, -1, 1],
            [-1, 0, 1, 1, -1, 1, 1, 1, -1],
            [-1, 1, 0, 1, -1, 1, 1, 1, -1],
            [-1, 1, 1, 0, -1, 1, 1, 1, -1],
            [1, -1, -1, -1, 0, -1, -1, -1, 1],
            [-1, 1, 1, 1, -1, 0, 1, 1, -1],
            [-1, 1, 1, 1, -1, 1, 0, 1, -1],
            [-1, 1, 1, 1, -1, 1, 1, 0, -1],
            [1, -1, -1, -1, 1, -1, -1, -1, 0]
        ], np.int64)
        #npt.assert_equal(network.weight_matrix, expected)
        print "Weight matrix:\n{0}".format(network.weight_matrix)

    @unittest.skip("Still working...")
    def test_init_storkey_2_exemplars(self):
        """
        Test the initialization of a Hopfield network using the Storkey Learning Rule
        and 2 exemplars
        """
        v_one = [1, -1, -1, -1, 1, -1, -1, -1, 1]
        v_two = [-1, -1, -1, 1, 1, 1, -1, -1, -1]
        network = HopfieldNetwork([v_one, v_two], learning_rule="Storkey")
        expected = np.array([
            [0, 0, 0, -2, 0, -2, 0, 0, 2],
            [0, 0, 2, 0, -2, 0, 2, 2, 0],
            [0, 2, 0, 0, -2, 0, 2, 2, 0],
            [-2, 0, 0, 0, 0, 2, 0, 0, -2],
            [0, -2, -2, 0, 0, 0, -2, -2, 0],
            [-2, 0, 0, 2, 0, 0, 0, 0, -2],
            [0, 2, 2, 0, -2, 0, 0, 2, 0],
            [0, 2, 2, 0, -2, 0, 2, 0, 0],
            [2, 0, 0, -2, 0, -2, 0, 0, 0]
        ], np.int64)
        # npt.assert_equal(network.weight_matrix, expected)
        print "Weight matrix is:\n{0}".format(network.weight_matrix)

    @unittest.skip("Still working...")
    def test_init_storkey_3_exemplars(self):
        """
        Test the initialization of a Hopfield network using 2 exemplars
        """
        v_one = [1, -1, -1, -1, 1, -1, -1, -1, 1]
        v_two = [-1, -1, -1, 1, 1, 1, -1, -1, -1]
        v_three = [-1, -1, 1, -1, -1, 1, -1, -1, 1]
        network = HopfieldNetwork([v_one, v_two, v_three], learning_rule="Storkey")
        expected = np.array([
            [0, 1, -1, -1, 1, -3, 1, 1, 1],
            [1, 0, 1, 1, -1, -1, 3, 3, -1],
            [-1, 1, 0, -1, -3, 1, 1, 1, 1],
            [-1, 1, -1, 0, 1, 1, 1, 1, -3],
            [1, -1, -3, 1, 0, -1, -1, -1, -1],
            [-3, -1, 1, 1, -1, 0, -1, -1, -1],
            [1, 3, 1, 1, -1, -1, 0, 3, -1],
            [1, 3, 1, 1, -1, -1, 3, 0, -1],
            [1, -1, 1, -3, -1, -1, -1, -1, 0]
        ])
        # npt.assert_equal(network.weight_matrix, expected)
        print "Weight matrix is:\n{0}".format(network.weight_matrix)

    @unittest.skip("Still working...")
    def test_recall(self):
        """
        Recall a reconstructed exemplar using a perfect exemplar
        """
        v_one = [1, -1, -1, -1, 1, -1, -1, -1, 1]
        v_two = [-1, -1, -1, 1, 1, 1, -1, -1, -1]
        network = HopfieldNetwork([v_one, v_two])

        # Check recall of exemplar 1
        results = network.recall([1, -1, -1, -1, 1, -1, -1, -1, 1])
        npt.assert_equal(results, v_one)

        # Check recall of exemplar 2
        results = network.recall([-1, -1, -1, 1, 1, 1, -1, -1, -1])
        npt.assert_equal(results, v_two)

    @unittest.skip("Still working...")
    def test_recall_noisy(self):
        """
        Recall a reconstructed exemplar using a noisy exemplar
        """
        v_one = [1, -1, -1, -1, 1, -1, -1, -1, 1]
        v_two = [-1, -1, -1, 1, 1, 1, -1, -1, -1]
        network = HopfieldNetwork([v_one, v_two])

        # single index change from v_one,  v_one[1] = -1
        p_noisy = [-1, -1, -1, -1, 1, -1, -1, -1, 1]
        p = network.recall(p_noisy)
        npt.assert_equal(v_one, p)

        # single index change to v_two
        p_noisy = [-1, -1, -1, 1, 1, 1, -1, 1, -1]
        p = network.recall(p_noisy)
        npt.assert_equal(v_two, p)

    @unittest.skip("Example from Module 8.3")
    def test_recall_noisy_module8(self):
        """
        Recall a reconstructed exemplar using a really noisy exemplar
        """
        v_one = [1, -1, -1, -1, 1, -1, -1, -1, 1]
        v_two = [-1, -1, -1, 1, 1, 1, -1, -1, -1]
        network = HopfieldNetwork([v_one, v_two])

        # noisy version of v_one
        p_noisy = [1, 1, -1, -1, -1, -1, -1, -1, 1]
        p1 = network.recall(p_noisy)
        p2 = network.recall(p1)
        results = network.recall(p2)
        npt.assert_equal(v_one, results)
        print "Synchronous recall results: {0}".format(results)

    @unittest.skip("Still working...")
    def test_asynchronous_recall(self):
        """
        Recall a reconstructed exemplar using the asynchronous method.
        """
        v_one = [1, -1, -1, -1, 1, -1, -1, -1, 1]
        v_two = [-1, -1, -1, 1, 1, 1, -1, -1, -1]
        network = HopfieldNetwork([v_one, v_two], debug=True)

        # Noisy version of exemplar 1
        p = [1, 1, -1, -1, -1, -1, -1, -1, 1]
        for i in range(0, 3):
            p = network.recall(p, asynchronous=True)
        npt.assert_equal(v_one, p)

    @unittest.skip("Still working...")
    def test_recall_storkey(self):
        """
        Recall an original exemplar using the actual original exemplar (no noise) in a network
        trained using the Storkey Learning rule
        """
        v_one = [1, -1, -1, -1, 1, -1, -1, -1, 1]
        v_two = [-1, -1, -1, 1, 1, 1, -1, -1, -1]
        network = HopfieldNetwork([v_one, v_two], learning_rule="Storkey")

        # Check recall of exemplar 1
        results = network.recall([1, -1, -1, -1, 1, -1, -1, -1, 1])
        npt.assert_equal(results, v_one)

        # Check recall of exemplar 2
        results = network.recall([-1, -1, -1, 1, 1, 1, -1, -1, -1])
        npt.assert_equal(results, v_two)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHopfieldNetwork)
    unittest.TextTestRunner(verbosity=2).run(suite)
