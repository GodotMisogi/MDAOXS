""" Testing out MPI optimization with pyopt_sparse"""

import os
import unittest
import numpy as np

from openmdao.api import IndepVarComp, ExecComp, LinearGaussSeidel, Component, \
    ParallelGroup, Problem, Group
from openmdao.core.mpi_wrap import MPI
from openmdao.test.mpi_util import MPITestCase
from openmdao.test.util import assert_rel_error, ConcurrentTestCaseMixin, \
                               set_pyoptsparse_opt
from openmdao.test.simple_comps import SimpleArrayComp
from openmdao.test.exec_comp_for_test import ExecComp4Test

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
    from openmdao.solvers.petsc_ksp import PetscKSP as lin_solver
    print('MPI USED')
else:
    from openmdao.core.basic_impl import BasicImpl as impl
    from openmdao.solvers.scipy_gmres import ScipyGMRES as lin_solver
    print("MPI NOT USED")

# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')


if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


class LinearSolver(Component):
    """Just a 1D Parabola."""

    def __init__(self, size=10, A=np.eye(10), b=np.arange(0,10)):

        super(LinearSolver, self).__init__()

        self.A = A
        self.b = b
        self.size = size
        # Params
        self.add_param('x', np.ones(shape=(size,)))

        # Unknowns
        self.add_state('obj', val=1.0)

    def solve_nonlinear(self, params, unknowns, resids):
        pass

    def apply_nonlinear(self, params, unknowns, resids):
        x = params['x']
        A = self.A
        b = self.b
        resids['obj'] = np.matmil(A,x)-b

    def linearize(self, params, unknowns, resids):
        """ derivs """
        J = {}
        J['obj', 'x'] = self.A
        return J



if __name__ == '__main__':
    #from openmdao.test.mpi_util import mpirun_tests
    #mpirun_tests()
    if OPT is None:
        raise unittest.SkipTest("pyoptsparse is not installed")

    if OPTIMIZER is None:
        raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

    size = 100
    prob = Problem(impl=impl)
    root = prob.root = Group()

    root.add('p', IndepVarComp('x', val=np.ones(shape=(size,))))
    root.add('c', LinearSolver(size=size, A=np.eye(size), b=np.arange(0, size)))
    root.connect('p.x', 'c.x')


    root.ln_solver = lin_solver()

    prob.setup()
    prob.run()
    err = np.sum(np.abs(prob['c.obj']))
    result = prob['p.x']
    print(result)
    print(err)