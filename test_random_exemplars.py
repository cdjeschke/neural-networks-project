import unittest

from random_exemplars import RandomExemplars


class TestRandomExemplars(unittest.TestCase):
    """
    Test generation of random exemplars
    """

    def test_generate_exemplars(self):
        exemplars = RandomExemplars.get_exemplars(4, 4)
        self.assertTrue(exemplars[0] == [-1, -1, -1, -1])
        self.assertTrue(exemplars[1] == [-1, 1, -1, -1])
        self.assertTrue(exemplars[2] == [1, -1, -1, -1])
        self.assertTrue(exemplars[3] == [1, 1, -1, -1])


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRandomExemplars)
    unittest.TextTestRunner(verbosity=2).run(suite)