"""This is script contains functions used to get from a quadrilateral
mesh to a Spectral-Element mesh. File for mesh object creation?"""


import numpy as np
import netCDF4

# Import GLL library to get the lagrange polynomials for interpolation
# of the grid
import gll_library


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
    connect = nc.variables['connect1']    
    
    return (np.array(X[:]),np.array(Y[:]),np.array(Z[:]),np.array(connect[:]))



def mesh_interp2D(X,Y,Z,connect,ngllx,nglly):
    """mesh_interp(X,Y,Z,connect,ngllx,nglly)

    Function takes in coordinates of meshgrid and its connectivity 
    matrix. Then, it interpolates the GLL points onto the global grid 
    and efines the numbering for the new found set of points.
    """
    
    # Number of elements from the number of rows of connectivity matrix
    nel,Nn = connect.shape

    # Loop over elements to interpolate GLL points 
    # for i in range(nel)
    
    # Safe shape function values in transform array
    tran

    # Counter to find indeces in 
    '''for i in range(ngllx):
        for j in range(nglly):
    '''
    




if __name__ == "__main__":
    """printing the connect matrix if called
    f the standard mesh
    """
    ngllx = 4
    ngllz = 4


    X,Y,Z,connect = readEx('../input/RectMesh.e')
    mesh_interp2D(X,Y,Z,connect,ngllx,ngllz)

    '''

   
    print(X)
    print(Z)
    print(connect)
    '''


