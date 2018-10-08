from petsc4py import PETSc
import petsc4py
from openmdao.api import ImplicitComponent,Problem, Group, IndepVarComp,DirectSolver,NonlinearRunOnce
from openmdao.api import PETScVector, PETScKrylov
import numpy as np
from mpi4py import MPI

class LnSolver(ImplicitComponent):
    def initialize(self):
        self.distributed = False
        self.options.declare('b',types=np.ndarray)
        self.options.declare('N')


    def setup(self):
        size =  self.options['N']
        self.add_input('A', val=np.eye(size))
        self.add_output('x', val=np.ones(size))
        rows = np.outer(np.arange(size), np.ones(size, int)).flatten()
        cols = np.arange(size ** 2)
        self.declare_partials('x','A', rows=rows, cols=cols)
        self.declare_partials('x','x')


    def solve_nonlinear(self, inputs, outputs):
        print("solving linear system")
        rank = PETSc.COMM_WORLD.rank
        num_ranks = PETSc.COMM_WORLD.size

        N = self.options['N']
        b = PETSc.Vec().createMPI(size=N, comm=MPI.COMM_WORLD)
        b.setValues(list(np.arange(N)), self.options['b'])
        b.assemble()
        A = PETSc.Mat()
        A.create(comm=MPI.COMM_WORLD)
        A.setSizes([N, N])
        A.setType("mpiaij")
        A.setUp()
        A.setValues([i for i in range(N)], [i for i in range(N)], inputs['A'])
        A.assemble()
        x, c = A.getVecs()
        x.set(0)
        ksp = PETSc.KSP()
        ksp.create(MPI.COMM_WORLD)
        ksp.setOperators(A)
        ksp.setFromOptions()
        ksp.solve(b, x)
        local_x = np.zeros(N)
        id_start, id_end = A.getOwnershipRange()
        # print("processors %d"%rank," owns %d,%d"%(id_start,id_end))
        local_x[id_start:id_end] = x[...]
        global_x = np.zeros(N)
        # print(np.array(x))
        MPI.COMM_WORLD.Reduce([local_x, MPI.DOUBLE], [global_x, MPI.DOUBLE], op=MPI.SUM, root=0)

        if rank == 0:
            #print(np.dot(inputs['A'],global_x)-self.options['b'])
            outputs['x'] = global_x

    def apply_nonlinear(self, inputs, outputs, residuals):
        b = self.options['b']
        residuals['x']=np.dot(inputs['A'],outputs['x'])-b

    def linearize(self, inputs, outputs, jacobian):
        size = self.options['N']
        jacobian['x', 'A'] = np.outer(np.ones(size), outputs['x']).flatten()
        jacobian['x', 'x'] = inputs['A']


if __name__ == '__main__':
    N = 500
    prob = Problem(model=Group())
    lhA = IndepVarComp()
    lhA.add_output(name='A',val=np.eye(N))

    prob.model.add_subsystem(name='lhA',subsys=lhA,promotes_outputs=['A'])
    prob.model.add_subsystem(name='ln',subsys=LnSolver(N=N, b=np.array(np.arange(N))))

    prob.model.linear_solver = DirectSolver()
    prob.model.nonlinear_solver = NonlinearRunOnce()
    #prob.model.nonlinear_solver.options['maxiter'] = 100
    #prob.model.nonlinear_solver.options['iprint'] = 0

    prob.setup()
    prob['A']=np.eye(N)
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
    import time
    t0 = time.clock()
    prob.run_model()
    t1 = time.clock()-t0

    print("linear solver using %d processors spend %f second"%(rank,t1))
    if rank == 0:

        print(prob['ln.x'])


