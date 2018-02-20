from __future__ import division
import numpy as np

from openmdao.api import ImplicitComponent
from scipy.sparse.linalg import gmres as gmres_solve

class FEM3dComp(ImplicitComponent):

    def initialize(self):
        """
        initialize function
        :return: void
        """

        """
        num_elements: total number of linear tetrahedral (solid) element
        shape = 1, dtype = int
        """
        self.metadata.declare('num_elements', types=int)
        self.metadata.declare('num_nodes', types=int)
        """
        num_node: total number of nodes
        """
        #self.metadata.declare('num_node',types=int)

        """
        mesh: FEM mesh information. For an unstructured mesh, it need some extra calculation (outside openmdao)
        shape = (element_ID, [x,y,z,nodeID], 4)
        """
        #self.metadata.declare('mesh')

        """
        ! involved into mesh
        node_positions: store the x,y,z value of each node in a row and sorted by ID
        shape = (total_node, 3), dtype = np.array
        loaded from a txt file
        """
        #self.metadata.declare('node_positions') # shape = (total_node,3)

        """
        ! involved into mesh
        adjacent_T: store the ID of 4 nodes in the same elements in a row  
        shape = (num_elements, 4), dtype = np.array 
        """
        # self.metadata.declare('adjacent_T')

        """
        E: modulus of elasticity
        shape = 1, val = 1, dtype = int
        """
        #self.metadata.declare('E')

        """
        NU: Poisson's ratio
        shape = 1, val = 1, dtype = int
        """
        #self.metadata.declare('NU')

    def setup(self):
        num_elements = self.metadata['num_elements']
        num_nodes = self.metadata['num_nodes']
        #E = self.metadata['E']
        #NU = self.metadata['NU']
        #mesh = self.metadata['mesh']
        #num_node = self.metadata['num_node']

        size = 3 * num_nodes
        """
        f: unassemembled force vector. Although we only have single-direction force, we assume it to be (fx,fy,fz) 
        shape = (3 * num_elements,1)
        """
        self.add_input('K', shape=(size, size))
        self.add_input('f', shape=size)
        self.add_output('d', shape=size)
        rows = np.outer(np.arange(size), np.ones(size, int)).flatten()
        cols = np.arange(size ** 2)
        self.declare_partials('d', 'K', rows=rows, cols=cols)
        self.declare_partials('d', 'd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['d'] = np.dot(inputs['K'], outputs['d']) - inputs['f']

    def linearize(self, inputs, outputs, partials):
        num_elements = self.metadata['num_elements']
        size = num_elements
        # self.lu = lu_factor(inputs['K'])
        self.K = inputs['K']
        partials['d', 'K'] = np.outer(np.ones(size), outputs['d']).flatten()
        partials['d', 'd'] = inputs['K']

    def solve_nonlinear(self, inputs, outputs):
        # self.lu = lu_factor(inputs['K'])
        # outputs['d'] = lu_solve(self.lu, inputs['f'])
        self.K = inputs['K']
        outputs['d'] = gmres_solve(inputs['K'], inputs['f'])[0]

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            # d_outputs['d'] = lu_solve(self.lu, d_residuals['d'], trans=0)
            d_outputs['d'] = gmres_solve(self.K, d_residuals['d'], trans=0)
        else:
            # d_residuals['d'] = lu_solve(self.lu, d_outputs['d'], trans=1)
            d_residuals['d'] = gmres_solve(self.K, d_outputs['d'], trans=1)




if __name__ == '__main__':
    pass