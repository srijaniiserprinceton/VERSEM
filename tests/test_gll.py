from .context import src
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
        xi, weights = src.gll_library.gll_pw(N)

        # Testing the Values
        np.testing.assert_array_almost_equal(xi,np.array([-1,0,1]))
        np.testing.assert_array_almost_equal(weights,np.array([1/3,4/3,1/3]))
    
    def testflattened_weights(self):
        """Testing flattened_weights() which is found in 
        src.gll_library. First test is for order 1 the second test for 
        the order 2.
        """

        ####### 1 ########
        # Setting Order
        N = 1
        
        # Getting GLL Points and weights
        xi, w_xi = src.gll_library.gll_pw(N)
        eta, w_eta = src.gll_library.gll_pw(N)
        
        # computing the flattened weights
        W = src.gll_library.flattened_weights2D(w_xi,w_eta)
        
        # Solution
        W_Sol = np.array([1,1,1,1])

        np.testing.assert_array_almost_equal(W,W_Sol)

        
        ####### 2 ########
        # Setting Order
        N = 2
        
        # Getting GLL Points and weights
        xi, w_xi = src.gll_library.gll_pw(N)
        eta, w_eta = src.gll_library.gll_pw(N)
        
        # computing the flattened weights
        W = src.gll_library.flattened_weights2D(w_xi,w_eta)
        
        # Solution
        W_Sol = np.array([1/9,4/9,1/9,4/9,16/9,4/9,1/9,4/9,1/9])

        np.testing.assert_array_almost_equal(W,W_Sol)



    def testLagrange(self):
        """ Tests lagrange() from the gll_library
        First Test is first order, second test is second order polynomial
        """
        
        ####### 1 ########
        # Setting Order
        N = 1
        
        # Getting GLL Points and weights
        xi, weights = src.gll_library.gll_pw(N)
        
        # Testing the output of lagrange
        self.assertEqual(src.gll_library.lagrange(0,-0.5,xi),0.75) 

        ####### 2 ########
        # Setting Order
        N = 2

        # Getting GLL Points and weights
        xi, weights = src.gll_library.gll_pw(N)
        
        # Testing the output of lagrange
        self.assertEqual(src.gll_library.lagrange(0,-0.5,xi),0.375)
        

    
    def testLagrange2D(self):
        """ Tests lagrange() from the gll_library
        First Test is first order, second test is second order polynomial
        """
        ###### 1 ########
        # Setting Order
        N = 1

        # Getting GLL Points and weights
        xi,  weights = src.gll_library.gll_pw(N)
        eta, weights = src.gll_library.gll_pw(N)
        
        # Testing the output of lagrange
        # coordinate is (x,y)=(-.5,.5) and degree of both is 1 and the
        # polynomial number is 0 of both: Should be 1 at (-1,-1) and 
        # -0.046875 at (-.5,5).
        self.assertEqual(src.gll_library.lagrange2D(0,-1,xi,0,-1,eta),1)
        self.assertEqual(src.gll_library.lagrange2D(1,-.5,xi,0,.5,eta),0.0625)



        ####### 2 ########
        # Setting Order
        N = 2

        # Getting GLL Points and weights
        xi,  weights = src.gll_library.gll_pw(N)
        eta, weights = src.gll_library.gll_pw(N)
        
        # Testing the output of lagrange
        # coordinate is (x,y)=(-.5,.5) and degree of both is 1 and the
        # polynomial number is 0 of both: Should be 1 at (-1,-1) and 
        # -0.046875 at (-.5,5).
        self.assertEqual(src.gll_library.lagrange2D(0,-1,xi,0,-1,eta),1)
        self.assertEqual(src.gll_library.lagrange2D(0,-.5,xi,0,.5,eta),-0.046875)
    
    def testLagrange1st1D(self):
        """Testing the 1D derivative of the lagrange polynomial
        computed by lagrange1st() in src/gll_library.py. First test is
        going to be of the first degree polynomial and second is going 
        to be of second degree
        """
        ###### 1 ######
        # Setting the order
        N = 1

        # Getting collocation points:
        xi,__ = src.gll_library.gll_pw(N)
        
        # Testing the output of the function
        self.assertEqual(src.gll_library.lagrange1st(0,0.5,xi),-0.5)
        self.assertEqual(src.gll_library.lagrange1st(0,1,xi),-0.5)
        self.assertEqual(src.gll_library.lagrange1st(1,0.25,xi),0.5)
        
        ###### 2 ######
        # Setting the order
        N = 2

        # Getting collocation points:
        xi,__ = src.gll_library.gll_pw(N)
        
        # Testing the output of the function
        self.assertEqual(src.gll_library.lagrange1st(0,0.5,xi),0)
        np.testing.assert_almost_equal(src.gll_library.lagrange1st(1,0.25,xi),-0.5)
        np.testing.assert_almost_equal(src.gll_library.lagrange1st(2,0.5,xi),1)
    
    def testLagrange1st2D(self):
        """Testing the 1D derivative of the lagrange polynomial in 2D.
        It is computed using the function lagrange1st2D() in 
        src/gll_library. The first test tests the polynomial of degree 1
        and the second test will test the polynomial of degree 2.
        """

        ###### 1 ######
        # Setting the order
        N = 1

        # Getting collocation points
        xi, __  = src.gll_library.gll_pw(N)
        eta, __ = src.gll_library.gll_pw(N)
        
        # Testing the values
        self.assertEqual(src.gll_library.lagrange1st2D(0,0,xi,0,0,eta,0),-0.25)
        self.assertEqual(src.gll_library.lagrange1st2D(0,1,xi,0,0,eta,0),-0.25)
        self.assertEqual(src.gll_library.lagrange1st2D(0,0.5,xi,0,0,eta,1),-0.125)
        self.assertEqual(src.gll_library.lagrange1st2D(0,1,xi,0,1,eta,1),0)

        ###### 2 ######
        # Setting the order
        N = 2

        # Getting collocation points
        xi, __  = src.gll_library.gll_pw(N)
        eta, __ = src.gll_library.gll_pw(N)
        
        # Testing the values
        self.assertEqual(src.gll_library.lagrange1st2D(0,0.5,xi,0,0,eta,0),0)
        self.assertEqual(src.gll_library.lagrange1st2D(0,1,xi,0,1,eta,0),0)
        self.assertEqual(src.gll_library.lagrange1st2D(0,1,xi,0,0,eta,0),0)
        self.assertEqual(src.gll_library.lagrange1st2D(0,1,xi,0,-1,eta,0),0.5)
        self.assertEqual(src.gll_library.lagrange1st2D(0,-1,xi,0,1,eta,1),0.5)
        self.assertEqual(src.gll_library.lagrange1st2D(1,0,xi,1,0,eta,1),0)
        self.assertEqual(src.gll_library.lagrange1st2D(2,0,xi,1,0,eta,0),0.5)
        self.assertEqual(src.gll_library.lagrange1st2D(2,0,xi,1,0,eta,1), 0)
        self.assertEqual(src.gll_library.lagrange1st2D(2,0,xi,2,0,eta,0),0)
        self.assertEqual(src.gll_library.lagrange1st2D(2,0,xi,2,0,eta,1),0)
    
    def testLagrangeDerMat2D(self):
        """Testing LagrangeDerMat2D() from gll_library
        First test tests the first order Polynomial and the second one 
        the second order polynomial.
        """
        
        ###### 1 ######
        # Setting the order
        N = 1

        # Getting collocation points
        xi,__ = src.gll_library.gll_pw(N)
        eta,__ = src.gll_library.gll_pw(N)
        
        # Solution:
        dN_Sol = np.array([[-0.25,0.25,-0.25,0.25],
                            [-0.25,-0.25,0.25,0.25]])
        # Print the derivative matrix
        dNdxi = src.gll_library.lagrangeDerMat2D(0,xi,0,eta)

        # Check whether correct:
        np.testing.assert_array_equal(dN_Sol,dNdxi)
        
        
        ###### 2 ######
        # Setting the order
        N = 2

        # Getting collocation points
        xi,__ = src.gll_library.gll_pw(N)
        eta,__ = src.gll_library.gll_pw(N)
        
        # Solution:     N:   1  2    3   4    5  6    7  8    9
        dN_Sol = np.array([[ 0, 0,   0, -0.5, 0, 0.5, 0, 0,   0],
                           [ 0,-0.5, 0, -0,   0, 0  , 0, 0.5, 0]])
        # Print the derivative matrix
        dNdxi = src.gll_library.lagrangeDerMat2D(0,xi,0,eta)

        # Check whether correct:
        np.testing.assert_array_equal(dN_Sol,dNdxi)
    

    def testdN_local2D(self):
        """Testing the dN derivative matrix evaluated at each gll point
        """

        ###### 1 ######
        # Setting the order
        N = 1

        # Getting collocation points
        xi,__ = src.gll_library.gll_pw(N)
        eta,__ = src.gll_library.gll_pw(N)
        
        # Solution:
        dN_Sol = np.array([[-0.5, 0.5, 0,   0],
                           [-0.5, 0,   0.5, 0]])
        # Print the derivative matrix
        dN = src.gll_library.dN_local_2D(xi,eta)

        # Check whether correct:
        np.testing.assert_array_almost_equal(dN_Sol,dN[0,:,:])

        




    def testJacobian2D(self):
        """Testing the Jacobian matrix multiplication Jacobian2D() from 
        src/gll_library.py. An arbitrary element setup is tested:
        
        ::

                y                    ( x , y )
                ^                     _     _
             3  |    __--4           | 0 , 0 |
             2  |  3-    |           | 3 , 1 |
             1  |/___----2           | 1 , 2 |
             0  1-----------> x      |_3 , 3_|  
                0  1  2  3
        


        """
 
        ###### 1 ######
        # Setting the order
        N = 1

        # Getting collocation points
        xi,__  = src.gll_library.gll_pw(N)
        eta,__ = src.gll_library.gll_pw(N)
        
        # Creating arbitrary coordinate matrix as shown in description
        x = np.array([[0,0],[3,1],[1,2],[3,3]])

        # Computing shape function derivative matrices
        xi1,eta1  = (-1,-1) # first node in reference element 
        dN1 = src.gll_library.lagrangeDerMat2D(xi1,xi,eta1,eta)
        
        # Computing shape function derivative matrices
        xi2,eta2  = (1,-1) # first node in reference element 
        dN2 = src.gll_library.lagrangeDerMat2D(xi2,xi,eta2,eta)

        # Computing shape function derivative matrices
        xi3,eta3  = (-1,1) # first node in reference element 
        dN3 = src.gll_library.lagrangeDerMat2D(xi3,xi,eta3,eta)
        
        # Computing shape function derivative matrices
        xi4,eta4  = (1,1) # first node in reference element 
        dN4 = src.gll_library.lagrangeDerMat2D(xi4,xi,eta4,eta)

        # Calculating Jacobian
        J1 = src.gll_library.Jacobian2D(dN1,x)
        J1_Sol = np.array([[1.5, 0.5],
                           [0.5, 1 ]])


        J2 = src.gll_library.Jacobian2D(dN2,x)
        J2_Sol = np.array([[1.5, 0.5],
                           [0,   1  ]])

        J3 = src.gll_library.Jacobian2D(dN3,x)
        J3_Sol = np.array([[1,   0.5],
                           [0.5, 1  ]] ) 

        J4 = src.gll_library.Jacobian2D(dN4,x)
        J4_Sol = np.array([[1, 0.5],
                           [0, 1  ]])

        # Testing
        np.testing.assert_array_equal(J1,J1_Sol)
        np.testing.assert_array_equal(J2,J2_Sol)
        np.testing.assert_array_equal(J3,J3_Sol)
        np.testing.assert_array_equal(J4,J4_Sol)

    def testGlobal_Derivative(self):
        """Testing the global derivative function with the same element
        setup as Jacobian test.

        
        ::

                y                    ( x , y )
                ^                     _     _
             3  |    __--4           | 0 , 0 |
             2  |  3-    |           | 3 , 1 |
             1  |/___----2           | 1 , 2 |
             0  1-----------> x      |_3 , 3_|  
                0  1  2  3
        


        """
 
        ###### 1 ######
        # Setting the order
        N = 1

        # Getting collocation points
        xi,__  = src.gll_library.gll_pw(N)
        eta,__ = src.gll_library.gll_pw(N)
        
        # Creating arbitrary coordinate matrix as shown in description
        x = np.array([[0,0],[3,1],[1,2],[3,3]])

        # Computing shape function derivative matrices
        xi1,eta1  = (-1,-1) # first node in reference element 
        dNdxi = src.gll_library.lagrangeDerMat2D(xi1,xi,eta1,eta)

        # Calculating Jacobian
        Jacob = src.gll_library.Jacobian2D(dNdxi,x)

        # Compute globale derivative
        dNdx = src.gll_library.global_derivative(Jacob,dNdxi)

        # Solution
        dNdx_Sol =np.array([[-0.2, 0.4, -0.2,0 ],
                            [-0.4,-0.2,  0.6,0 ]])

        np.testing.assert_array_almost_equal(dNdx,dNdx_Sol)



    def testLegendre1D(self):
        """Testing legendre() from the gll_library
        First test tests the first order polynomial and the second one
        the second order polynomial.
        """

        ###### 1 ######
        # Setting the order
        N = 1

        # Getting collocation points
        xi,__ = src.gll_library.gll_pw(N)
        
        # Testing the ouput og the legendre polynomial of degree 1
        self.assertEqual(src.gll_library.legendre(0,0.25,xi),-1-1/3)
        self.assertEqual(src.gll_library.legendre(1,0.5   ,xi),2/3)

        ###### 2 ######
        # Setting the order
        N = 2

        # Getting collocation points
        xi,__ = src.gll_library.gll_pw(N)
        print(xi)
        # Testing the ouput og the legendre polynomial of degree 2
        self.assertEqual(src.gll_library.legendre(0,-0.5,xi),-2-2/3)
        self.assertEqual(src.gll_library.legendre(1,-0.5   ,xi),2-2/3)
        self.assertEqual(src.gll_library.legendre(2,0.5   ,xi),2+2/3)

if __name__ == "__main__":
    unittest.main()
