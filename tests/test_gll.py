from context import src
import unittest
import numpy as np




class TestGLL(unittest.TestCase):
    """ Testing functions of the gll_library 
    - Gauss-Lobatto-Legendre points and weights
    - Lagrange Polynomial evaluator"""

    def testGLLPointsAndWeights2(self):
        """ Tests gll_pw() from gll_library whether points and weights 
        are correct. Note that it is almost equal due to the fact that
        the points are hardcoded.
        """
        
        # Setting Order
        N = 2

        # Getting Points and Weights
        xi, weights = src.gll_pw(N)

        # Testing the Values
        np.testing.assert_array_almost_equal(xi,np.array([-1,0,1]))
        np.testing.assert_array_almost_equal(weights,np.array([1/3,4/3,1/3]))

    
    def testLagrangePoly(self):
        """ Tests lagrange() from the gll_library"""
        
        # Setting Order
        N = 2

        # Getting GLL Points and weights
        xi, weights = src.gll_pw(N)
        
        # Testing the output of lagrange
        self.assertEqual(src.lagrange(0,-0.5,xi),0.75)



if __name__ == "__main__":
    unittest.main()
