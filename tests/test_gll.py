from context import src
import unittest
import numpy as np


class TestGLL(unittest.TestCase):
    """ Testing functions concerning Gauss-Lobatto-Legendre
    points and weights"""

    def testGLLPointsAndWeights2(self):
        """ Checking whether points and weights are almost correct
        """
        
        # Getting Points and Weights
        xi, weights = src.gll_pw(2)

        # Testing the Values
        np.testing.assert_array_almost_equal(xi,np.array([-1,0,1]))
        np.testing.assert_array_almost_equal(weights,np.array([1/3,4/3,1/3]))


