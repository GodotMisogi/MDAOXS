from petsc4py import PETSc
import numpy as np
N = 7
A = PETSc.Mat()
A.create()
A.setSizes([N, N])
A.setType('aij')
A.setUp()
Temp = np.eye(N)
Temp[0,-1] = 1
row = range(N-1)
col = range(N-1)
data = np.arange(N)
for idx,(i,j) in enumerate(zip(row,col)):
    A.setValue(i,j,data[idx])
A.assemble()

B = A.convert('dense')
print(A[-1,-1])