
# importing necessary packages:
import netCDF4
import numpy as np


# Loading the Dataset
nc = netCDF4.Dataset('../input/RectMesh.e')


# Getting coordinates
X = nc.variables['coordx']
Y = nc.variables['coordy']
Z = nc.variables['coordz']

# Getting global coordinate numbering
connect = nc.variables['connect1']

# Print coordinates

print(X[:],Z[:],connect[:])

# Plotting the mesh
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib

xy = np.array([X[:], Z[:]]).T

patches = []
for coords in xy[connect[:]-1]:
    quad = Polygon(coords, True)
    patches.append(quad)

fig, ax = plt.subplots()
colors = 100 * np.random.rand(len(patches))
p = PatchCollection(patches, cmap=matplotlib.cm.coolwarm, alpha=0.4,edgecolor='k')
p.set_array(np.array(colors))
ax.add_collection(p)
ax.set_xlim([0, 20 ])
ax.set_ylim([0, -10])
ax.set_aspect('equal')
ax.invert_yaxis()
plt.show()

nc.close()
