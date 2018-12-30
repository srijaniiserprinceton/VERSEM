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

    
    def testLagrange(self):
        """ Tests lagrange() from the gll_library
        First Test is first order, second test is second order polynomial
        """
        
        ####### 1 ########
        # Setting Order
        N = 1
        
        # Getting GLL Points and weights
        xi, weights = src.gll_pw(N)
        
        # Testing the output of lagrange
        self.assertEqual(src.lagrange(0,-0.5,xi),0.75) 

        ####### 2 ########
        # Setting Order
        N = 2

        # Getting GLL Points and weights
        xi, weights = src.gll_pw(N)
        
        # Testing the output of lagrange
        self.assertEqual(src.lagrange(0,-0.5,xi),0.375)
        

    
    def testLagrange2D(self):
        """ Tests lagrange() from the gll_library
        First Test is first order, second test is second order polynomial
        """
        ###### 1 ########
        # Setting Order
        N = 1

        # Getting GLL Points and weights
        xi,  weights = src.gll_pw(N)
        eta, weights = src.gll_pw(N)
        
        # Testing the output of lagrange
        # coordinate is (x,y)=(-.5,.5) and degree of both is 1 and the
        # polynomial number is 0 of both: Should be 1 at (-1,-1) and 
        # -0.046875 at (-.5,5).
        self.assertEqual(src.lagrange2D(0,-1,xi,0,-1,eta),1)
        self.assertEqual(src.lagrange2D(1,-.5,xi,0,.5,eta),0.0625)



        ####### 2 ########
        # Setting Order
        N = 2

        # Getting GLL Points and weights
        xi,  weights = src.gll_pw(N)
        eta, weights = src.gll_pw(N)
        
        # Testing the output of lagrange
        # coordinate is (x,y)=(-.5,.5) and degree of both is 1 and the
        # polynomial number is 0 of both: Should be 1 at (-1,-1) and 
        # -0.046875 at (-.5,5).
        self.assertEqual(src.lagrange2D(0,-1,xi,0,-1,eta),1)
        self.assertEqual(src.lagrange2D(0,-.5,xi,0,.5,eta),-0.046875)

    
    def testLegendre1D(self):
        """Testing legendre() form the gll_library
        First test tests the first order polynomial and the second one
        the second order polynomial.
        """

        ###### 1 ######
        # Setting the order
        N = 1

        # Getting collocation points
        xi,__ = src.gll_pw(N)
        print(xi)
        # Testing the ouput og the legendre polynomial of degree 1
        self.assertEqual(src.legendre(0,0.25,xi),-1-1/3)
        self.assertEqual(src.legendre(1,0.5   ,xi),2/3)

        ###### 2 ######
        # Setting the order
        N = 2

        # Getting collocation points
        xi,__ = src.gll_pw(N)
        print(xi)
        # Testing the ouput og the legendre polynomial of degree 2
        self.assertEqual(src.legendre(0,-0.5,xi),-2-2/3)
        self.assertEqual(src.legendre(1,-0.5   ,xi),2-2/3)
        self.assertEqual(src.legendre(2,0.5   ,xi),2+2/3)

if __name__ == "__main__":
    unittest.main()
