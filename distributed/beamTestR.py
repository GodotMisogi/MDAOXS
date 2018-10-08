from __future__ import division
import numpy as np
from openmdao.api import ExplicitComponent,Group, IndepVarComp,Problem,ScipyOptimizeDriver,ImplicitComponent


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
        self.declare_partials('I', 'h', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        b = self.options['b']

        outputs['I'] = 1./12. * b * inputs['h'] ** 3

    def compute_partials(self, inputs, partials):
        b = self.options['b']

        partials['I', 'h'] = 1./4. * b * inputs['h'] ** 2



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
        coeffs[0, :] = [12, 6 * L0, -12, 6 * L0]
        coeffs[1, :] = [6 * L0, 4 * L0 ** 2, -6 * L0, 2 * L0 ** 2]
        coeffs[2, :] = [-12, -6 * L0, 12, -6 * L0]
        coeffs[3, :] = [6 * L0, 2 * L0 ** 2, -6 * L0, 4 * L0 ** 2]
        coeffs *= E / L0 ** 3

        self.mtx = mtx = np.zeros((num_elements, 4, 4, num_elements))
        for ind in range(num_elements):
            self.mtx[ind, :, :, ind] = coeffs

        self.declare_partials('K_local', 'I',
            val=self.mtx.reshape(16 * num_elements, num_elements))

    def compute(self, inputs, outputs):
        num_elements = self.options['num_elements']

        outputs['K_local'] = 0
        for ind in range(num_elements):
            outputs['K_local'][ind, :, :] = self.mtx[ind, :, :, ind] * inputs['I'][ind]

from six.moves import range
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


class StatesComp(ImplicitComponent):

    def initialize(self):
        self.options.declare('num_elements', types=int)
        self.options.declare('force_vector', types=np.ndarray)

    def setup(self):
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        size = 2 * num_nodes - 2

        self.add_input('K_local', shape=(num_elements, 4, 4))
        self.add_output('d', shape=size)

        cols = np.arange(16*num_elements)
        rows = np.repeat(np.arange(4), 4)
        rows = np.tile(rows, num_elements) + np.repeat(np.arange(num_elements), 16) * 2

        self.declare_partials('d', 'K_local', rows=rows, cols=cols)
        self.declare_partials('d', 'd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        force_vector = self.options['force_vector'][2:]

        self.K = self.assemble_CSC_K(inputs)
        residuals['d'] = np.dot(self.K, outputs['d']) - force_vector

    def solve_nonlinear(self, inputs, outputs):
        force_vector = self.options['force_vector'][2:]

        self.K = self.assemble_CSC_K(inputs)
        self.lu = splu(self.K)

        outputs['d'] = self.lu.solve(force_vector)

    def linearize(self, inputs, outputs, jacobian):
        num_elements = self.options['num_elements']

        self.K = self.assemble_CSC_K(inputs)
        self.lu = splu(self.K)

        i_elem = np.tile(np.arange(4), 4)
        i_d = np.tile(i_elem, num_elements) + np.repeat(np.arange(num_elements), 16) * 2

        jacobian['d', 'K_local'] = outputs['d'][i_d]

        jacobian['d', 'd'] = self.K.toarray()

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['d'] = self.lu.solve(d_residuals['d'])
        else:
            d_residuals['d'] = self.lu.solve(d_outputs['d'])

    def assemble_CSC_K(self, inputs):
        """
        Assemble the stiffness matrix in sparse CSC format.

        Returns
        -------
        ndarray
            Stiffness matrix as dense ndarray.
        """
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        num_entry = num_elements * 12 + 4
        ndim = num_entry

        data = np.zeros((ndim, ))
        cols = np.empty((ndim, ))
        rows = np.empty((ndim, ))

        # First element.
        data[:16] = inputs['K_local'][0, :, :].flat
        cols[:16] = np.tile(np.arange(4), 4)
        rows[:16] = np.repeat(np.arange(4), 4)

        j = 16
        for ind in range(1, num_elements):
            ind1 = 2 * ind
            K = inputs['K_local'][ind, :, :]

            # NW quadrant gets summed with previous connected element.
            data[j-6:j-4] += K[0, :2]
            data[j-2:j] += K[1, :2]

            # NE quadrant
            data[j:j+4] = K[:2, 2:].flat
            rows[j:j+4] = np.array([ind1, ind1, ind1 + 1, ind1 + 1])
            cols[j:j+4] = np.array([ind1 + 2, ind1 + 3, ind1 + 2, ind1 + 3])

            # SE and SW quadrants together
            data[j+4:j+12] = K[2:, :].flat
            rows[j+4:j+12] = np.repeat(np.arange(ind1 + 2, ind1 + 4), 4)
            cols[j+4:j+12] = np.tile(np.arange(ind1, ind1 + 4), 2)

            j += 12
        for idx,(row,col) in enumerate(zip(rows,cols)):
            row = row - 2
            col = col - 2




        n_K = 2 * num_nodes - 2
        return coo_matrix((data, (rows, cols)), shape=(n_K, n_K)).tocsc()



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
        self.declare_partials('displacements', 'd', val=1., rows=arange, cols=arange)

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
                              val=force_vector.reshape((1, 2 * num_nodes)))

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

        self.declare_partials('volume', 'h', val=b * L0)

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
            'states_comp.K_local')
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

num_elements = 50

prob = Problem(model=BeamGroup(E=E, L=L, b=b, volume=volume, num_elements=num_elements))

prob.driver = ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-9
prob.driver.options['disp'] = True

prob.setup()

prob.run_driver()

print(prob['inputs_comp.h'])