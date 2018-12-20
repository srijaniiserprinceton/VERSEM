"""This is script contains functions used to get from a quadrilateral
mesh to a Spectral-Element mesh. File for mesh object creation?"""


import matplotlib.pyplot as plt
import numpy as np
import netCDF4


# Import GLL library to get the lagrange polynomials for interpolation
# of the grid
import gll_library as gll


def readEx(name):
    """readEx(name)
    Function reads Exodus file and outputs Coordinates, connection 
    matrix, boundary coordinates and nodes.
    """
    
    # Loading the Dataset
    nc = netCDF4.Dataset(name) 
    
    # Getting coordinates
    X = nc.variables['coordx']
    Y = nc.variables['coordy']
    Z = nc.variables['coordz']

    # Getting global coordinate numbering                                   
    connect = np.array(nc.variables['connect1'])-1    
    
    return (np.array(X[:]),np.array(Y[:]),np.array(Z[:]),np.array(connect[:]))



def mesh_interp2D(X,Y,Z,connect,ngllx,nglly):
    """mesh_interp(X,Y,Z,connect,ngllx,nglly)

    Function takes in coordinates of meshgrid and its connectivity 
    matrix. Then, it interpolates the GLL points onto the global grid 
    and efines the numbering for the new found set of points.
    """
    pass
    # Number of elements (nel) from the number of rows of connectivity 
    # matrix and number of control points (Nn) from the columns
    nel,Nn = connect.shape
    
    # Number of points per side
    Nside = int(np.round(np.sqrt(Nn)))

    # Local control points
    xi_control ,__ = gll.gll_pw(Nside-1) # -1 since gll_pw takes in polynomial degree


    # Node setup in mesh file
    #
    #      3----4               3--7--4
    #      |    |               |  |  |  ??
    #      |    |               6--9--8
    #      2----1               |  |  |
    #                           2--5--1
    #
    # Local number of the lagrange polynomials for the control points
    # entirely dependent on numbering of the 
    if Nn == 4:
        polynum  = np.array([[1,0],[0,0],[0,1],[1,1]])
    elif Nn == 9:
        polynum  = np.array([[2,0],[0,0],[0,2],[2,2],\
                            [1,0],[0,1],[1,2],[2,1],[1,1]])
         
    
    # Save shape function values in transform array
    gll_shape_matrix = np.zeros([ngllx*nglly,Nn])

    # GLL Points
    xi ,__ = gll.gll_pw(ngllx-1)  # points in x direction to be interpolated
    eta ,__ = gll.gll_pw(nglly-1) # points in y direction to be interpolated

    print(xi)
    print(eta)
    
    # Total number of refined GLL points
    NN = ngllx*nglly
    
    # Counter to find local indeces
    index = np.zeros([NN,1])

    # Loop over local GLL points
    # GLL indexing and coordinates
    local_ind   = np.zeros([NN,1])
    local_coord = np.zeros([NN,2])
    
    counter = 0
    for j in range(nglly):
        for i in range(ngllx):
            # GLL point locations:
            local_coord[counter,:] = np.array([xi[i],eta[j]])
            counter+=1

    for i in range(NN):
        # Calculate each control point's shape function at each GLL
        # point
        for k in range(Nn):
            gll_shape_matrix[i,k] = \
                    gll.lagrange2D( polynum[k,0],local_coord[i,0],xi_control,\
                                    polynum[k,1],local_coord[i,1],xi_control)
   
                    
    print(gll_shape_matrix)
    
    #######
    # Find global GLL points and numbering for EVERY element:

    # Empty connectivity array
    gll_connect = np.zeros([nel,NN])
    tol = 1e-12
    
    for i in range(nel):
        

        # Calculate global GLL coordinates for 1 element:
        glob_gll = np.array([gll_shape_matrix @ X[connect[0,:]].T, \
                             gll_shape_matrix @ Z[connect[0,:]].T]).T
        
        
        if i == 1: 
            # Create new element coordinates with NN number of nodes
            gll_coordinates   = glob_gll
            gll_connect[i,:]  = np.arange(0,NN)
        
        else    
                
            for j in range(NN):
                """Looping over variable to check whether there is 
                overlap between the elements"""
                
                # Check whether coordinate exists already
                ind = np.where(\
                        np.abs(np.gll_coordinates[:,0]-glob_gll[j,0])<tol && \
                        np.abs(np.gll_coordinates[:,1]-glob_gll[j,1])<tol)
                
                # if it doesnt exist
                if ind.size==0
                    # append new coordinates to global data set 
                    np.append(gll_coordinates, glob_gll[j,:],axis=0)
                    
                    # numbering at highest recent number +1
                    gll_connect[i,j] = np.amax(gll_connect) + 1
                    
                # if it already exists
                else
                    # do NOT append a new coordinate, but use index for 
                    # numbering purpose
                    gll_connect[i,j] = ind

        
                

    print(glob_gll)
    print(connect[0,:])        
    print(X)
    print(Z)
    print(connect)
    plt.figure(1)
    plt.scatter(glob_gll[:,0],glob_gll[:,1], marker='x')
    plt.scatter(X[connect[0,:]],Z[connect[0,:]],marker='o')
    plt.show()

    plt.figure(2)
    plt.




if __name__ == "__main__":
    """printing the connect matrix if called
    f the standard mesh
    """
    ngllx = 4
    ngllz = 4


    X,Y,Z,connect = readEx('../input/RectMesh.e')
    print(X)
    print(Z)
    print(connect)
    
    mesh_interp2D(X,Y,Z,connect,ngllx,ngllz)

    

   



