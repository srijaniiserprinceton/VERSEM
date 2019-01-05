import numpy as np
import gll_library as gll
import loc2glob as l2g
#######################################################################
###            Constructing Element Mass Matrices                  ####
#######################################################################

def el_mass_mat(ngll_el,rho,J,W):
    """.. function:: el_mass_mat(ngll_total,rho,J,W)

    Computes the Elemetal Mass Matrix M_e for each element

    :param ngll_el: total number of gll points in an element

    :param rho: flattened density matrix [1]x[total Number of GLL points]
              (``numpy``)

    :param J: flattened Jacobian determinant matrix [1]x[total Number of GLL points]
              (``numpy``)

    :param W: flattened weight matrix [1]x[total Number of GLL points]
              (``numpy``)
    
    :rtype: ``numpy`` [ngll_total]x[ngll_total] array

    The description of the Jacobian can be found on the theory
    documentation.
    """

    #the element mass matrix
    M = np.zeros([ngll_el,ngll_el])
    for i in range(ngll_el):
        M[i,i] = rho[i]*J[i]*W[i]

    return M

def glob_el_mass_mat(el_no,ngll_el,gll_coordinates,gll_connect,rho,dN_local,W):
    """.. function:: glob_el_mass_mat(el_no,ngll_total,gll_coordinates,gll_connect,dN_local,W)

    Computes the Global Mass Matrix Mg

    :param el_no: total number of elements in the domain

    :param ngll_el: total number of gll points in an element

    :param rho: flattened density matrix [1]x[ngll_el]
              (``numpy``)

    :param W: flattened weight matrix [1]x[ngll_el]
              (``numpy``)
    
    :rtype: ``numpy`` [ngll_total]x[ngll_total] array

    """
    Mg = np.zeros([len(gll_coordinates),len(gll_coordinates)])
    J_el = np.zeros(len(gll_coordinates))
    for i in range(el_no):    #el_no is the total number of elements
        gll_coords_el = gll_coordinates[gll_connect[i]]
        for j in range(len(gll_coords_el)):
            #Jacobian for the nodes in a specific element
    	    J_el[j] = np.linalg.det(gll.Jacobian2D(dN_local[j,:,:],gll_coords_el))

        #We are now ready to construct the element matrices
        Me = el_mass_mat(ngll_el,rho,J_el,W)
        #Constructing global mass matrix
        Mg += l2g.local2global(Me,Mg,gll_connect,[i])
    return Mg
    
