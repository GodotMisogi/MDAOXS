from petsc4py import PETSc
import petsc4py
from openmdao.api import ImplicitComponent,Problem, Group, IndepVarComp,ExplicitComponent
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


    def compute(self, inputs, outputs):
        b = PETSc.Vec().createMPI(size=N, comm=PETSc.COMM_WORLD)
        print('b')
        b.setValues(list(np.arange(N)), self.options['b'])
        b.assemble()
        A = PETSc.Mat()
        A.create(comm=PETSc.COMM_WORLD)
        A.setSizes([N, N])
        A.setType("mpiaij")
        A.setUp()
        A.setValues([i for i in range(N)], [i for i in range(N)], inputs['A'])
        A.assemble()
        x,c = A.getVecs()
        x.set(0)
        y = np.zeros(N)
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)
        ksp.setOperators(A)
        ksp.setFromOptions()
        ksp.solve(b, x)
        comm = MPI.COMM_WORLD
        rank = MPI.COMM_WORLD.Get_rank()
        comm.Reduce([x[:], MPI.DOUBLE],[y, MPI.DOUBLE],op=MPI.SUM, root=0)

        if rank == 0:
            print(np.dot(inputs['A'],y)-b)

        outputs['x'] = y


    def apply_nonlinear(self, inputs, outputs, residuals):
        b = self.options['b']
        residuals['x']=np.dot(inputs['A'],outputs['x'])-b

    def linearize(self, inputs, outputs, jacobian):
        size = self.options['N']
        jacobian['x', 'A'] = np.outer(np.ones(size), outputs['x']).flatten()
        jacobian['x', 'x'] = inputs['A']


if __name__ == '__main__':
    N = 10
    prob = Problem(model=Group())
    lhA = IndepVarComp()
    lhA.add_output(name='A',val=np.eye(N))

    prob.model.add_subsystem(name='lhA',subsys=lhA,promotes_outputs=['A'])
    prob.model.add_subsystem(name='ln',subsys=LnSolver(N=N, b=np.array(np.arange(N))))

    #prob.model.linear_solver = PETScKrylov()
    #prob.model.nonlinear_solver = NewtonSolver()
    #prob.model.nonlinear_solver.options['maxiter'] = 100
    #prob.model.nonlinear_solver.options['iprint'] = 0

    prob.setup()
    prob['A']=np.eye(N)
    prob.run_model()
    print(prob['ln.x'])