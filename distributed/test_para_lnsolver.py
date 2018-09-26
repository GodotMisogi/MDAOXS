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
else:
    from openmdao.core.basic_impl import BasicImpl as impl
    from openmdao.solvers.scipy_gmres import ScipyGMRES as lin_solver

# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')


if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


class LinearSolver(Component):
    """Just a 1D Parabola."""

    def __init__(self, A=np.eye(10), b=np.arange(0,10)):

        super(LinearSolver, self).__init__()

        self.A = A
        self.b = b
        self.size = np.size(b)
        # Params
        self.add_param('x', np.ones(shape=(self.size,)))

        # Unknowns
        self.add_output('y', np.ones(shape=(self.size,)))

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much. """
        unknowns['y'] = self.A*params['x'] - self.b

    def linearize(self, params, unknowns, resids):
        """ derivs """
        J = {}
        J['y', 'x'] = self.A
        return J


class MP_Point(Group):

    def __init__(self, size=10):
        super(MP_Point, self).__init__()
        self.add('p', IndepVarComp('x', val=np.ones(shape=(size,))))
        self.add('c', LinearSolver(A=np.eye(size),b=np.arange(0,size)))
        self.connect('p.x', 'c.x')


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
    root.ln_solver = lin_solver()
    root.add('p', IndepVarComp('x', val=np.ones(shape=(size,))))
    root.add('c', LinearSolver(A=np.eye(size), b=np.arange(0, size)))
    root.connect('p.x', 'c.x')

    root.add('total', ExecComp('obj = np.sum(y*y)'))
    root.connect('c.y','total.y')

    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = OPTIMIZER
    prob.driver.options['print_results'] = False
    prob.driver.add_desvar('x', lower=-1.0, upper=size + 1.0)

    prob.driver.add_objective('total.obj')

    prob.root.ln_solver.options['mode'] = 'rev'
    prob.run()
    err = prob['total.obj']
    result = prob['x']
    print(result)