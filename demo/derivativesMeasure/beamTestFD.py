from __future__ import division
import numpy as np
from openmdao.api import IndepVarComp, ExplicitComponent, Group, ImplicitComponent, Problem, ScipyOptimizeDriver,PETScVector
from scipy.linalg import lu_factor, lu_solve
import scipy.io as si
class MomentOfInertiaComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_elements', types=int)
        self.options.declare('b')

    def setup(self):
        num_elements = self.options['num_elements']

        self.add_input('h', shape=num_elements)
        self.add_output('I', shape=num_elements)

        rows = np.arange(num_elements)
        cols = np.arange(num_elements)
        self.declare_partials('I', 'h', method='fd')

    def compute(self, inputs, outputs):
        b = self.options['b']

        outputs['I'] = 1./12. * b * inputs['h'] ** 3

class LocalStiffnessMatrixComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_elements', types=int)
        self.options.declare('E')
        self.options.declare('L')

    def setup(self):
        num_elements = self.options['num_elements']
        E = self.options['E']
        L = self.options['L']

        self.add_input('I', shape=num_elements)
        self.add_output('K_local', shape=(num_elements, 4, 4))

        L0 = L / num_elements
        coeffs = np.empty((4, 4))
        coeffs[0, :] = [12 , 6 * L0, -12, 6 * L0]
        coeffs[1, :] = [6 * L0, 4 * L0 ** 2, -6 * L0, 2 * L0 ** 2]
        coeffs[2, :] = [-12 , -6 * L0, 12, -6 * L0]
        coeffs[3, :] = [6 * L0, 2 * L0 ** 2, -6 * L0, 4 * L0 ** 2]
        coeffs *= E / L0 ** 3

        self.mtx = mtx = np.zeros((num_elements, 4, 4, num_elements))
        for ind in range(num_elements):
            self.mtx[ind, :, :, ind] = coeffs

        self.declare_partials('K_local', 'I', method='fd')

    def compute(self, inputs, outputs):
        num_elements = self.options['num_elements']

        outputs['K_local'] = 0
        for ind in range(num_elements):
            outputs['K_local'][ind, :, :] = self.mtx[ind, :, :, ind] * inputs['I'][ind]

class GlobalStiffnessMatrixComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_elements', types=int)

    def setup(self):
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1

        self.add_input('K_local', shape=(num_elements, 4, 4))
        self.add_output('K', shape=(2 * num_nodes + 2, 2 * num_nodes + 2))

        rows = np.zeros(16 * num_elements, int)
        indices = np.arange(
            ((2 * num_nodes + 2) * (2 * num_nodes + 2))
        ).reshape((2 * num_nodes + 2, 2 * num_nodes + 2))
        ind1, ind2 = 0, 0
        for ind in range(num_elements):
            ind2 += 16
            ind1_ = 2 * ind
            ind2_ = 2 * ind + 4
            rows[ind1:ind2] = indices[ind1_:ind2_, ind1_:ind2_].flatten()
            ind1 += 16
        cols = np.arange(16 * num_elements)
        self.declare_partials('K', 'K_local', rows=rows, cols=cols, val=1.)

        self.set_check_partial_options('K_local', step=1e0)

    def compute(self, inputs, outputs):
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1

        outputs['K'][:, :] = 0.
        for ind in range(num_elements):
            ind1_ = 2 * ind
            ind2_ = 2 * ind + 4

            outputs['K'][ind1_:ind2_, ind1_:ind2_] += inputs['K_local'][ind, :, :]
        outputs['K'][2 * num_nodes + 0, 0] = 1.0
        outputs['K'][2 * num_nodes + 1, 1] = 1.0
        outputs['K'][0, 2 * num_nodes + 0] = 1.0
        outputs['K'][1, 2 * num_nodes + 1] = 1.0

class StatesComp(ImplicitComponent):

    def initialize(self):
        self.options.declare('num_elements', types=int)
        self.options.declare('force_vector', types=np.ndarray)

    def setup(self):
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        size = 2 * num_nodes + 2

        self.add_input('K', shape=(size, size))
        self.add_output('d', shape=size)

        rows = np.outer(np.arange(size), np.ones(size, int)).flatten()
        cols = np.arange(size ** 2)
        self.declare_partials('d', 'K', rows=rows, cols=cols)

        self.declare_partials('d', 'd', method='fd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        force_vector = np.concatenate([self.options['force_vector'], np.zeros(2)])

        residuals['d'] = np.dot(inputs['K'], outputs['d']) - force_vector

    def solve_nonlinear(self, inputs, outputs):
        print('solving linear system')
        force_vector = np.concatenate([self.options['force_vector'], np.zeros(2)])
        self.lu = lu_factor(inputs['K'])
        outputs['d'] = lu_solve(self.lu, force_vector)

    def linearize(self, inputs, outputs, partials):
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        size = 2 * num_nodes + 2

        partials['d', 'K'] = np.outer(np.ones(size), outputs['d']).flatten()
        partials['d', 'd'] = inputs['K']

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['d'] = lu_solve(self.lu, d_residuals['d'], trans=0)
        else:
            d_residuals['d'] = lu_solve(self.lu, d_outputs['d'], trans=1)

class DisplacementsComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_elements', types=int)

    def setup(self):
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        size = 2 * num_nodes + 2

        self.add_input('d', shape=size)
        self.add_output('displacements', shape=2 * num_nodes)

        arange = np.arange(2 * num_nodes)
        self.declare_partials('displacements', 'd', rows=arange, cols=arange, val=1.)

    def compute(self, inputs, outputs):
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1

        outputs['displacements'] = inputs['d'][:2 * num_nodes]
class ComplianceComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_elements', types=int)
        self.options.declare('force_vector', types=np.ndarray)

    def setup(self):
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        force_vector = self.options['force_vector']

        self.add_input('displacements', shape=2 * num_nodes)
        self.add_output('compliance')

        self.declare_partials('compliance', 'displacements',
            method='fd')

    def compute(self, inputs, outputs):
        force_vector = self.options['force_vector']

        outputs['compliance'] = np.dot(force_vector, inputs['displacements'])

class VolumeComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_elements', types=int)
        self.options.declare('b', default=1.)
        self.options.declare('L')

    def setup(self):
        num_elements = self.options['num_elements']
        b = self.options['b']
        L = self.options['L']
        L0 = L / num_elements

        self.add_input('h', shape=num_elements)
        self.add_output('volume')

        self.declare_partials('volume', 'h', method='fd')

    def compute(self, inputs, outputs):
        num_elements = self.options['num_elements']
        b = self.options['b']
        L = self.options['L']
        L0 = L / num_elements

        outputs['volume'] = np.sum(inputs['h'] * b * L0)

class BeamGroup(Group):

    def initialize(self):
        self.options.declare('E')
        self.options.declare('L')
        self.options.declare('b')
        self.options.declare('volume')
        self.options.declare('num_elements', int)

    def setup(self):
        E = self.options['E']
        L = self.options['L']
        b = self.options['b']
        volume = self.options['volume']
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1

        force_vector = np.zeros(2 * num_nodes)
        force_vector[-2] = -1.

        inputs_comp = IndepVarComp()
        inputs_comp.add_output('h', shape=num_elements)
        self.add_subsystem('inputs_comp', inputs_comp)

        I_comp = MomentOfInertiaComp(num_elements=num_elements, b=b)
        self.add_subsystem('I_comp', I_comp)

        comp = LocalStiffnessMatrixComp(num_elements=num_elements, E=E, L=L)
        self.add_subsystem('local_stiffness_matrix_comp', comp)

        comp = GlobalStiffnessMatrixComp(num_elements=num_elements)
        self.add_subsystem('global_stiffness_matrix_comp', comp)

        comp = StatesComp(num_elements=num_elements, force_vector=force_vector)
        self.add_subsystem('states_comp', comp)

        comp = DisplacementsComp(num_elements=num_elements)
        self.add_subsystem('displacements_comp', comp)

        comp = ComplianceComp(num_elements=num_elements, force_vector=force_vector)
        self.add_subsystem('compliance_comp', comp)

        comp = VolumeComp(num_elements=num_elements, b=b, L=L)
        self.add_subsystem('volume_comp', comp)

        self.connect('inputs_comp.h', 'I_comp.h')
        self.connect('I_comp.I', 'local_stiffness_matrix_comp.I')
        self.connect(
            'local_stiffness_matrix_comp.K_local',
            'global_stiffness_matrix_comp.K_local')
        self.connect(
            'global_stiffness_matrix_comp.K',
            'states_comp.K')
        self.connect(
            'states_comp.d',
            'displacements_comp.d')
        self.connect(
            'displacements_comp.displacements',
            'compliance_comp.displacements')
        self.connect(
            'inputs_comp.h',
            'volume_comp.h')

        self.add_design_var('inputs_comp.h', lower=1e-2, upper=10.)
        self.add_objective('compliance_comp.compliance')
        self.add_constraint('volume_comp.volume', equals=volume)

E = 1.
L = 1.
b = 0.1
volume = 0.01

num_elements = 10

# prob.driver = ScipyOptimizeDriver()
# prob.driver.options['optimizer'] = 'SLSQP'
# prob.driver.options['tol'] = 1e-9
# prob.driver.options['disp'] = True

from contextlib import contextmanager
import time
@contextmanager
def timeblock ( label ):
    start = time.clock()
    try:
        yield
    finally:
        end = time.clock()
        print ('{}:{} '. format (label , end - start ))


for num_elements in [500]:
    prob = Problem(model=BeamGroup(E=E, L=L, b=b, volume=volume, num_elements=num_elements))

    prob.setup(distributed_vector_class=PETScVector)
    with timeblock('fd time cost with ' + str(num_elements)+ " elements: "):
        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = True
        prob.driver.options['maxiter'] = 0
        prob.run_driver()
        #dd = prob.compute_totals(wrt=['inputs_comp.h'], of=['compliance_comp.compliance'])

#h = prob['inputs_comp.h']
#import matplotlib.pyplot as plt
#print(dd)
#plt.plot(np.arange(len(h)),h)
#plt.show()
#prob.check_totals(compact_print=True)