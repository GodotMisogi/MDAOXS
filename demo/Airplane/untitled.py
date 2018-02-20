import numpy as np
from scipy.sparse import *

def changeAijtoV(list,i,j,V):
    list[i,j] = V


list_ = np.array([[1,2,3],[2,3,4]])
changeAijtoV(list_,0,0,5)

list_[0,0]

csr_matrix(shape=(1000,1000))