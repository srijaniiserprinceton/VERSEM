from .context import src
import unittest
import numpy as np

class testLoc2Glob(unittest.TestCase):

    def test(self):
        Me = np.ones([4,4])
        Mg = np.zeros([8,8])

        node_num = np.array([[4,1,2,5],[5,2,3,6],[7,4,5,8]])-1
        el_no = np.array(range(0,len(node_num)))
 
        for i in el_no:
            Mg += src.loc2glob.local2global(Me,Mg,node_num,[i])

        Mg_sol = np.array([[1., 1., 0., 1., 1., 0., 0., 0.],
                           [1., 2., 1., 1., 2., 1., 0., 0.],
                           [0., 1., 1., 0., 1., 1., 0., 0.],
                           [1., 1., 0., 2., 2., 0., 1., 1.],
                           [1., 2., 1., 2., 3., 1., 1., 1.],
                           [0., 1., 1., 0., 1., 1., 0., 0.],
                           [0., 0., 0., 1., 1., 0., 1., 1.],
                           [0., 0., 0., 1., 1., 0., 1., 1.]])

        np.testing.assert_array_almost_equal(Mg,Mg_sol)

        ########################################################

        Me = np.ones([4,4])
        Mg = np.zeros([6,6])

        node_num = np.array([[1,2,3,4],[2,5,4,6]])-1
        el_no = np.array(range(0,len(node_num)))

        for i in el_no:
            Mg += src.local2global(Me,Mg,node_num,[i])

        Mg_Sol = np.array([[1., 1., 1., 1., 0., 0.],
                           [1., 2., 1., 2., 1., 1.],
                           [1., 1., 1., 1., 0., 0.],
                           [1., 2., 1., 2., 1., 1.],
                           [0., 1., 0., 1., 1., 1.],
                           [0., 1., 0., 1., 1., 1.]])
        np.testing.assert_array_almost_equal(Mg,Mg_Sol)

if __name__ == "__main__":
    unittest.main()
