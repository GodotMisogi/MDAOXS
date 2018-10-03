from openmdao.api import PETScVector
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

rank = MPI.COMM_WORLD.Get_rank()
num_ranks = MPI.COMM_WORLD.Get_size()

N = 20
b = PETScVector().createMPI(size=N,comm=MPI.COMM_WORLD)
b.setValues(list(np.arange(N)), np.arange(N))
b.assemble()
A = PETSc.Mat()
A.create(comm=PETSc.COMM_WORLD)
A.setSizes([N, N])
A.setType("mpiaij")
A.setUp()
Temp = np.eye(N)
Temp[0,-1] = 1
A.setValues([i for i in range(N)], [i for i in range(N)], Temp)


A.assemble()
ksp = PETSc.KSP()
ksp.create(PETSc.COMM_WORLD)
ksp.setOperators(A)
ksp.setFromOptions()
x = PETSc.Vec().createMPI(N)
ksp.solve(b, x)
y = np.zeros(N)
MPI.COMM_WORLD.Reduce([x[:],MPI.DOUBLE],[y,MPI.DOUBLE],op=MPI.SUM,root=0)
print(x[:])
if rank == 0:
    print(y)