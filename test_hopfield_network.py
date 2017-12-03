import numpy as np
import numpy.testing as npt
import unittest

from hopfield_network import HopfieldNetwork


class TestHopfieldNetwork(unittest.TestCase):
    """
    Unit test for retrieving the exemplars to use in assignment 2
    """

    @unittest.skip("Working on it")
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

    @unittest.skip("Working on it")
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

    @unittest.skip("Working on it")
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

    def test_perfect_recall_hebbian(self):
        """
        Recall a reconstructed exemplar using a perfect exemplar
        """
        v_one = [1, -1, -1, -1, 1, -1, -1, -1, 1]
        v_two = [-1, -1, -1, 1, 1, 1, -1, -1, -1]
        network = HopfieldNetwork([v_one, v_two], learning_rule="Hebb")

        # Check recall of exemplar 1 under async & sync
        results = network.synchronous_recall([1, -1, -1, -1, 1, -1, -1, -1, 1])
        p = results[-1]
        npt.assert_equal(p, v_one)
        results = network.asynchronous_recall([1, -1, -1, -1, 1, -1, -1, -1, 1])
        p = results[-1]
        npt.assert_equal(p, v_one)

        # Check recall of exemplar 2 under async & sync
        results = network.synchronous_recall([-1, -1, -1, 1, 1, 1, -1, -1, -1])
        p = results[-1]
        npt.assert_equal(p, v_two)
        results = network.synchronous_recall([-1, -1, -1, 1, 1, 1, -1, -1, -1])
        p = results[-1]
        npt.assert_equal(p, v_two)


    @unittest.skip("Example from Module 8.3. Won't recall correctly under synchronous modality.")
    def test_noisy_synchronous_recall_hebbian(self):
        """
        Recall a reconstructed exemplar using the asynchronous method.
        """
        v_one = [1, -1, -1, -1, 1, -1, -1, -1, 1]
        v_two = [-1, -1, -1, 1, 1, 1, -1, -1, -1]
        network = HopfieldNetwork([v_one, v_two], learning_rule="Hebb")

        # Noisy version of exemplar 1
        v_p = [1, 1, -1, -1, -1, -1, -1, -1, 1]
        results = network.synchronous_recall(v_p)
        p = results[-1]
        npt.assert_equal(p, v_one)

    def test_noisy_asynchronous_recall_hebbian(self):
        """
        Recall a reconstructed exemplar using the asynchronous method.
        """
        v_one = [1, -1, -1, -1, 1, -1, -1, -1, 1]
        v_two = [-1, -1, -1, 1, 1, 1, -1, -1, -1]
        network = HopfieldNetwork([v_one, v_two], learning_rule="Hebb")

        # two index change from v_one,  v_one[1] = -1, v_one[4] = -1
        v_p = [1, 1, -1, -1, -1, -1, -1, -1, 1]
        results = network.asynchronous_recall(v_p)
        p = results[-1]
        npt.assert_equal(p, v_one)

        # single index change to v_two
        v_p = [-1, -1, -1, 1, 1, 1, -1, 1, -1]
        results = network.asynchronous_recall(v_p)
        p = results[-1]
        npt.assert_equal(p, v_two)

    # Start of tests for Storkey Learning Rule

    def test_init_storkey(self):
        """
        Test that the weights matrix for a Hopfield Network trained using the Storkey Learning Rule
        is calculated correctly.
        """
        # Single exemplar
        expected_weights = np.array([
            [0, .3333333, .3333333],
            [.3333333, 0, .3333333],
            [.3333333, .3333333, 0]
        ])

        v_one = [1, 1, 1]
        network = HopfieldNetwork([v_one], learning_rule="Storkey")
        npt.assert_almost_equal(network.weight_matrix, expected_weights)

        # 2 exemplars
        expected_weights = np.array([
            [0, 0, .8888888],
            [0, 0, 0],
            [.8888888, 0, 0]
        ])
        v_one = [1, 1, 1]
        v_two = [1, -1, 1]
        network = HopfieldNetwork([v_one, v_two], learning_rule="Storkey")
        npt.assert_almost_equal(network.weight_matrix, expected_weights)

    def test_perfect_recall_storkey(self):
        """
        Recall an original exemplar using the actual original exemplar (no noise) in a network
        trained using the Storkey Learning rule
        """
        v_one = [1, -1, -1, -1, 1, -1, -1, -1, 1]
        v_two = [-1, -1, -1, 1, 1, 1, -1, -1, -1]
        v_three = [-1, -1, 1, -1, -1, 1, -1, -1, 1]
        network = HopfieldNetwork([v_one, v_two, v_three], learning_rule="Storkey")

        # Check recall of exemplar 1 - sync & async
        results = network.synchronous_recall([1, -1, -1, -1, 1, -1, -1, -1, 1])
        p = results[-1]
        npt.assert_equal(p, v_one)
        results = network.asynchronous_recall([1, -1, -1, -1, 1, -1, -1, -1, 1])
        p = results[-1]
        npt.assert_equal(p, v_one)

        # Check recall of exemplar 2
        results = network.synchronous_recall([-1, -1, -1, 1, 1, 1, -1, -1, -1])
        p = results[-1]
        npt.assert_equal(p, v_two)
        results = network.asynchronous_recall([-1, -1, -1, 1, 1, 1, -1, -1, -1])
        p = results[-1]
        npt.assert_equal(p, v_two)

        # Check recall of exemplar 3
        results = network.synchronous_recall([-1, -1, 1, -1, -1, 1, -1, -1, 1])
        p = results[-1]
        npt.assert_equal(p, v_three)
        results = network.asynchronous_recall([-1, -1, 1, -1, -1, 1, -1, -1, 1])
        p = results[-1]
        npt.assert_equal(p, v_three)

    def test_noisy_recall_storkey(self):
        """
        Recall an original exemplar using the actual original exemplar (no noise) in a network
        trained using the Storkey Learning rule
        """
        v_one = [1, -1, -1, -1, 1, -1, -1, -1, 1]
        v_two = [-1, -1, -1, 1, 1, 1, -1, -1, -1]
        v_three = [-1, -1, 1, -1, -1, 1, -1, -1, 1]
        network = HopfieldNetwork([v_one, v_two, v_three], learning_rule="Storkey")

        # Check recall of exemplar 1
        results = network.synchronous_recall([1, -1, -1, -1, 1, -1, -1, -1, 1])
        p = results[-1]
        npt.assert_equal(p, v_one)
        results = network.synchronous_recall([1, -1, -1, -1, 1, -1, -1, -1, 1])
        p = results[-1]
        npt.assert_equal(p, v_one)

        # Check recall of exemplar 2
        results = network.synchronous_recall([-1, -1, -1, 1, 1, 1, -1, -1, -1])
        p = results[-1]
        npt.assert_equal(p, v_two)
        results = network.asynchronous_recall([-1, -1, -1, 1, 1, 1, -1, -1, -1])
        p = results[-1]
        npt.assert_equal(p, v_two)

        # Check recall of exemplar 3
        results = network.synchronous_recall([1, -1, 1, -1, -1, 1, -1, 1, 1])
        p = results[-1]
        npt.assert_equal(p, v_three)
        results = network.synchronous_recall([1, -1, 1, -1, -1, 1, -1, 1, 1])
        p = results[-1]
        npt.assert_equal(p, v_three)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHopfieldNetwork)
    unittest.TextTestRunner(verbosity=2).run(suite)
