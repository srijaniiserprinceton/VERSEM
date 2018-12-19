"""This is script contains functions used to get from a quadrilateral
mesh to a Spectral-Element mesh. File for mesh object creation?"""


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
    connect = nc.variables['connect1']    
    
    return (np.array(X[:]),np.array(Y[:]),np.array(Z[:]),np.array(connect[:]))



def mesh_interp2D(X,Y,Z,connect,ngllx,nglly):
    """mesh_interp(X,Y,Z,connect,ngllx,nglly)

    Function takes in coordinates of meshgrid and its connectivity 
    matrix. Then, it interpolates the GLL points onto the global grid 
    and efines the numbering for the new found set of points.
    """
    pass
    # Number of elements from the number of rows of connectivity matrix
    nel,Nn = connect.shape
    
    # Number of points per side
    Nside = np.round(np.sqrt(Nn))

    # Local control points
    xi_control ,__ = gll.gll_pw(Nside-1) # -1 since gll_pw takes in polynomial degree


    # Node setup in mesh file 
    #      3----4
    #      |    |
    #      |    |
    #      2----1
    # Local number of the lagrange polynomials for the control points
    # entirely dependent on numbering of the 
    if Nn == 4
        polynum  = np.array([[1,0],[0,0],[0,1],[1,1]])
    elif Nn == 9
        polynum  = np.array([[2,0],[0,0],[0,2],[2,2],\
                            [1,0],[0,1],[1,2],[2,1],[1,1]])
         
    
    # Save shape function values in transform array
    gll_shape_matrix = np.zeros(ngllx,nglly,Nn)


    # GLL Points
    xi ,__ = gll.gll_pw(Nside-1) # -1 since gll_pw takes in polynomial degree
    eta ,__ = gll.gll_pw(Nside-1) # -1 since gll_pw takes in polynomial degree

    # Counter to find local indeces
    index = np.zeros(ngllx*nglly,1)

    # Loop over local GLL points
    # The loop order just switch for simplified numbering later on
    # primary index is x overarching y
    for j in range(nglly):
        for i in range(ngllx):
            
            # Calculate each control point's shape function at each GLL
            # point
            for k in range(np.sqrt(Nn)):
                gll_shape_matrix[i,j,k] = \
                        gll.lagrange2D( polynum[k,0],xi[i],xi_control,\
                                        polynum[k,1],eta[j],xi_control)

    # Calculate GLL Points for the first element:
    

            
    




if __name__ == "__main__":
    """printing the connect matrix if called
    f the standard mesh
    """
    ngllx = 4
    ngllz = 4


    X,Y,Z,connect = readEx('../input/RectMesh.e')
    #mesh_interp2D(X,Y,Z,connect,ngllx,ngllz)

    

   
    print(X)
    print(Z)
    print(connect)


