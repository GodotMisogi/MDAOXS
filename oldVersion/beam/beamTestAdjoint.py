from __future__ import division
import numpy as np
from openmdao.api import IndepVarComp, Component, Group, Problem, ScipyOptimizer
from scipy.linalg import lu_factor, lu_solve
import scipy.io as si
class MomentOfInertiaComp(Component):

    def initialize(self):
        self.metadata.declare('num_elements', types=int)
        self.metadata.declare('b')

    def setup(self):
        num_elements = self.metadata['num_elements']

        self.add_param('h', val=np.ones(shape=(num_elements,)))
        self.add_output('I', val=np.ones(shape=(num_elements,)))

        rows = np.arange(num_elements)
        cols = np.arange(num_elements)
        self.declare_partials('I', 'h', rows=rows, cols=cols)
    def solve_nonlinear(self, params, unknowns, resids):
        b = self.metadata['b']

        unknowns['I'] = 1./12. * b * params['h'] ** 3

    def linearize(self, params, unknowns, resids):
        J = {}
        b = self.metadata['b']

        J['I', 'h'] = 1. / 4. * b * params['h'] ** 2
        return J


class LocalStiffnessMatrixComp(Component):

    def initialize(self):
        self.metadata.declare('num_elements', types=int)
        self.metadata.declare('E')
        self.metadata.declare('L')

    def setup(self):
        num_elements = self.metadata['num_elements']
        E = self.metadata['E']
        L = self.metadata['L']

        self.add_param('I', val=np.ones(shape=(num_elements,)))
        self.add_output('K_local', val=np.ones(shape=(num_elements, 4, 4)))

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

        self.declare_partials('K_local', 'I',
            val=self.mtx.reshape(16 * num_elements, num_elements))

    def solve_nonlinear(self, params, unknowns, resids):
        num_elements = self.metadata['num_elements']

        unknowns['K_local'] = 0
        for ind in range(num_elements):
            unknowns['K_local'][ind, :, :] = self.mtx[ind, :, :, ind] * params['I'][ind]

class GlobalStiffnessMatrixComp(Component):

    def initialize(self):
        self.metadata.declare('num_elements', types=int)

    def setup(self):
        num_elements = self.metadata['num_elements']
        num_nodes = num_elements + 1

        self.add_param('K_local', val=np.ones(shape=(num_elements, 4, 4)))
        self.add_output('K', val=np.ones(shape=(2 * num_nodes + 2, 2 * num_nodes + 2)))

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
        self.declare_partials('K', 'K_local', val=1., rows=rows, cols=cols)

    def linearize(self, params, unknowns, resids):
        num_elements = self.metadata['num_elements']
        num_nodes = num_elements + 1

        unknowns['K'][:, :] = 0.
        for ind in range(num_elements):
            ind1_ = 2 * ind
            ind2_ = 2 * ind + 4

            unknowns['K'][ind1_:ind2_, ind1_:ind2_] += params['K_local'][ind, :, :]

        unknowns['K'][2 * num_nodes + 0, 0] = 1.0
        unknowns['K'][2 * num_nodes + 1, 1] = 1.0
        unknowns['K'][0, 2 * num_nodes + 0] = 1.0
        unknowns['K'][1, 2 * num_nodes + 1] = 1.0

class StatesComp(Component):

    def initialize(self):
        self.metadata.declare('num_elements', types=int)
        self.metadata.declare('force_vector', types=np.ndarray)

    def setup(self):
        num_elements = self.metadata['num_elements']
        num_nodes = num_elements + 1
        size = 2 * num_nodes + 2

        self.add_param('K', val=np.ones(shape=(size, size)))
        self.add_output('d', val=np.ones(shape=(size,)))

        rows = np.outer(np.arange(size), np.ones(size, int)).flatten()
        cols = np.arange(size ** 2)
        self.declare_partials('d', 'K', rows=rows, cols=cols)

        self.declare_partials('d', 'd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        force_vector = np.concatenate([self.metadata['force_vector'], np.zeros(2)])

        residuals['d'] = np.dot(inputs['K'], outputs['d']) - force_vector

    def _sys_solve_nonlinear(self, params, unknowns, resids):
        force_vector = np.concatenate([self.metadata['force_vector'], np.zeros(2)])

        self.lu = lu_factor(params['K'])

        unknowns['d'] = lu_solve(self.lu, force_vector)
    def linearize(self, params, unknowns, resids):
        J = {}
        num_elements = self.metadata['num_elements']
        num_nodes = num_elements + 1
        size = 2 * num_nodes + 2

        self.lu = lu_factor(params['K'])

        J['d', 'K'] = np.outer(np.ones(size), unknowns['d']).flatten()
        J['d', 'd'] = params['K']
        return J

    def solve_linear(self, dumat, drmat, vois, mode=None):
        if mode == 'fwd':
            unknowns['d'] = lu_solve(self.lu, resids['d'], trans=0)
        else:
            resids['d'] = lu_solve(self.lu, d_outputs['d'], trans=1)

class DisplacementsComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_elements', types=int)

    def setup(self):
        num_elements = self.metadata['num_elements']
        num_nodes = num_elements + 1
        size = 2 * num_nodes + 2

        self.add_input('d', shape=size)
        self.add_output('displacements', shape=2 * num_nodes)

        arange = np.arange(2 * num_nodes)
        self.declare_partials('displacements', 'd', val=1., rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        num_elements = self.metadata['num_elements']
        num_nodes = num_elements + 1

        outputs['displacements'] = inputs['d'][:2 * num_nodes]
class ComplianceComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_elements', types=int)
        self.metadata.declare('force_vector', types=np.ndarray)

    def setup(self):
        num_elements = self.metadata['num_elements']
        num_nodes = num_elements + 1
        force_vector = self.metadata['force_vector']

        self.add_input('displacements', shape=2 * num_nodes)
        self.add_output('compliance')

        self.declare_partials('compliance', 'displacements',
            val=force_vector.reshape((1, 2 * num_nodes)))

    def compute(self, inputs, outputs):
        force_vector = self.metadata['force_vector']

        outputs['compliance'] = np.dot(force_vector, inputs['displacements'])

class VolumeComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_elements', types=int)
        self.metadata.declare('b', default=1.)
        self.metadata.declare('L')

    def setup(self):
        num_elements = self.metadata['num_elements']
        b = self.metadata['b']
        L = self.metadata['L']
        L0 = L / num_elements

        self.add_input('h', shape=num_elements)
        self.add_output('volume')

        self.declare_partials('volume', 'h', val=b * L0)

    def compute(self, inputs, outputs):
        num_elements = self.metadata['num_elements']
        b = self.metadata['b']
        L = self.metadata['L']
        L0 = L / num_elements

        outputs['volume'] = np.sum(inputs['h'] * b * L0)

class BeamGroup(Group):

    def initialize(self):
        self.metadata.declare('E')
        self.metadata.declare('L')
        self.metadata.declare('b')
        self.metadata.declare('volume')
        self.metadata.declare('num_elements', int)

    def setup(self):
        E = self.metadata['E']
        L = self.metadata['L']
        b = self.metadata['b']
        volume = self.metadata['volume']
        num_elements = self.metadata['num_elements']
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



E = 1E6
L = 1.
b = 0.1
volume = 0.01

num_elements = 50

prob = Problem(model=BeamGroup(E=E, L=L, b=b, volume=volume, num_elements=num_elements))

prob.driver = ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-9
prob.driver.options['disp'] = True

prob.setup()
prob.run_driver()

K = prob['states_comp.K']

print(np.linalg.det(K))
"""
import test.timing
for num_elements in [10, 100, 200,500, 1000, 2000]:
    prob = Problem(model=BeamGroup(E=E, L=L, b=b, volume=volume, num_elements=num_elements))

    # prob.driver = ScipyOptimizeDriver()
    # prob.driver.options['optimizer'] = 'SLSQP'
    # prob.driver.options['tol'] = 1e-9
    # prob.driver.options['disp'] = True

    prob.setup()
    with test.timing.timeblock('adjoint time cost with ' + str(num_elements)+ " elements: "):
        dd = prob.compute_totals(wrt=['inputs_comp.h'], of=['compliance_comp.compliance'])
"""
h = prob['inputs_comp.h']
print(h)
d = prob['displacements_comp.d']
print(d[::2])
#import matplotlib.pyplot as plt
#print(dd)
#plt.plot(np.arange(len(h)),h)
#plt.show()
#prob.check_totals(compact_print=True)