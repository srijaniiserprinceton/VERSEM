from .context import src
import unittest
import numpy as np

Me = np.ones([4,4])
Mg = np.zeros([8,8])

node_num = np.array([[4,1,2,5],[5,2,3,6],[7,4,5,8]])-1
el_no = np.array(range(0,len(node_num)))

for i in el_no:
    Mg += src.loc2glob.local2global(Me,Mg,node_num,[i])

print(Mg)

