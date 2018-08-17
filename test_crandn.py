import unittest
import numpy as np
from numpy.testing import (
    assert_equal, assert_almost_equal,
    assert_array_equal, assert_array_almost_equal
    )
from comm_utils import crandn

class TestCrandn(unittest.TestCase):

    def setUp(self):
        self.seed = 9394

    def test_output_size(self):
        np.random.seed(self.seed)
        actual = crandn(2,3).shape
        desired = (2,3)
        assert_equal(actual, desired)

    def test_output_array(self):
        np.random.seed(self.seed)
        actual = crandn(2,4)
        desired = np.array([[ 6.7167935206112228e-01 + 0.0290943289914996j,
                             -2.9786895888987697e-01 + 0.3063631170489249j,
                             -2.6839965045618325e-01 + 0.07325852363382324j,
                              4.8972392673274601e-01 + 0.13290225159683072j],
                            [ 1.3552479337020603e-03 + 1.626731293907899j,
                              1.6136582896925378e-01 - 0.5487080631154108j,
                              2.5487425553794667e+00 + 0.6261762976429994j,
                              4.1799291238980468e-01 - 0.5461132562044057j]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_variance(self):
        np.random.seed(self.seed)
        actual = np.var(crandn(100))
        desired = 1.088794935637427
        assert_almost_equal(actual, desired, decimal=15)


if __name__ == '__main__':
    unittest.main()
